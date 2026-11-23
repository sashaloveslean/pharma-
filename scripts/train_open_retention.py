#!/usr/bin/env python3
"""Evaluate open RT measurements with molecule and system disjoint tests."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from open_retention import number, load_observations


@lru_cache(maxsize=100000)
def molecular(smiles):
    mol = Chem.MolFromSmiles(smiles)
    desc = [f(mol) for f in (Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA,
            Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.NumRotatableBonds)]
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128).GetFingerprint(mol)
    return desc + list(fp)


def vector(row, numeric):
    values = molecular(row["smiles"]) + [
        number(row["metadata"].get(k)) if number(row["metadata"].get(k)) is not None else np.nan
        for k in numeric
    ]
    # Full program remains in observations. Sample channel schedules at fixed times,
    # and preserve duration, to distinguish gradients from static initial %B.
    points = row["gradient_program"]
    times = [number(p.get("t [min]")) for p in points]
    if points and all(t is not None for t in times) and times == sorted(times):
        values += [max(times)]
        for channel in ["A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]:
            concentrations = [number(p.get(channel)) for p in points]
            values += list(np.interp([0, 1, 3, 5, 10, 15, 20, 30, 60], times, concentrations)) if all(
                v is not None for v in concentrations) else [np.nan] * 9
    else:
        values += [np.nan] * 46
    values += [row["metadata"].get("column.name") or "unknown",
               row["metadata"].get("column.usp.code") or "unknown"]
    return values


def metrics(actual, predicted):
    error = np.abs(actual - predicted)
    return {"mae_min": float(np.mean(error)), "median_absolute_error_min": float(np.median(error))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/open_retention/observations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("models/open_retention"))
    p.add_argument("--mode", default="RP", choices=["RP", "HILIC"])
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--max-rows", type=int, help="Deterministic training sample; all source data retained")
    args = p.parse_args()
    rows = load_observations(args.data)
    rows = [r for r in rows if r["source_info"] and r["source_info"][0].get("method.type") == args.mode]
    available_rows = len(rows)
    if args.max_rows and len(rows) > args.max_rows:
        indices = sorted(np.random.default_rng(42).choice(len(rows), args.max_rows, replace=False))
        rows = [rows[i] for i in indices]
    if len(rows) < 20 or len({r["compound_id"] for r in rows}) < 3:
        raise SystemExit("Need >=20 observations and >=3 molecules in the selected mode")
    # No source IDs, t0 or response-derived fields as predictors.
    numeric = sorted({k for r in rows for k, v in r["metadata"].items()
                      if k not in {"id", "column.t0"} and number(v) is not None})
    X = np.asarray([vector(r, numeric) for r in rows], dtype=object)
    y = np.asarray([r["retention_time_min"] for r in rows])
    systems = np.asarray([r["system_id"] for r in rows])
    def provenance_group(row):
        info = row["source_info"][0]
        if row["system_id"].startswith("uliege:"):
            return ":".join(row["system_id"].split(":")[:2])
        if "smrt" in str(info.get("name", "")).lower() or "Domingo-Almenara" in str(info.get("source", "")):
            return "publication:METLIN_SMRT"
        return info.get("url", "").strip().lower().rstrip("/") or row["system_id"]
    system_groups = np.asarray([provenance_group(r) for r in rows])
    # Connectivity grouping also blocks leakage through stereo variants.
    molecules = np.asarray([r["compound_id"].split("-")[0] for r in rows])
    params = dict(iterations=args.iterations, depth=5, learning_rate=0.06,
                  random_seed=42, verbose=False, allow_writing_files=False,
                  thread_count=4, cat_features=[X.shape[1]-2, X.shape[1]-1])
    report = {"rows": len(rows), "available_rows": available_rows, "molecules": len(set(molecules)), "systems": len(set(systems)),
              "mode": args.mode, "source_revisions": sorted({r["source_revision"] for r in rows}),
              "target": "log1p(tR_min)", "numeric_metadata_features": numeric,
              "limitations": ["No validated prediction on customer's instrument",
                              "No peak widths or confirmed co-injection: no Rs forecast",
                              "No instrument dwell volume correction",
                              "Some source URLs are absent; remaining cross-source duplicates need curation",
                              "Gradient sampled at fixed times; original program retained"],
              "validation": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    predictions = []
    tests = [("new_molecule", GroupKFold(n_splits=min(5, len(set(molecules)))), molecules)]
    if len(set(system_groups)) >= 3:
        tests.append(("new_system", LeaveOneGroupOut() if len(set(system_groups)) <= 10 else GroupKFold(n_splits=5), system_groups))
    else:
        report["validation"]["new_system"] = {"status": "not_evaluable", "reason": "Fewer than three independent system provenance groups"}
    report["system_holdout_grouping"] = "source URL/publication; known SMRT aliases grouped; U Liege by physical instrument"
    for name, splitter, groups in tests:
        pred, baseline = np.empty_like(y), np.empty_like(y)
        folds = []
        for train, test in splitter.split(X, y, groups):
            model = CatBoostRegressor(**params)
            model.fit(X[train], np.log1p(y[train]))
            pred[test] = np.maximum(0, np.expm1(model.predict(X[test])))
            by_system = {s: np.median(y[train][systems[train] == s]) for s in set(systems[train])}
            baseline[test] = [by_system.get(systems[i], np.median(y[train])) for i in test]
            folds.append({"test_rows": len(test), **metrics(y[test], pred[test])})
        # Bootstrap groups rather than treating correlated rows as independent.
        rng = np.random.default_rng(42)
        errors = [np.abs(y[groups == g] - pred[groups == g]) for g in sorted(set(groups))]
        boot = [np.mean(np.concatenate([errors[i] for i in rng.integers(0, len(errors), len(errors))]))
                for _ in range(300)]
        report["validation"][name] = {
            **metrics(y, pred), "baseline": metrics(y, baseline), "folds": folds,
            "splitter": type(splitter).__name__,
            "mae_group_bootstrap_95_interval_min": list(np.quantile(boot, [0.025, 0.975])),
            "per_system": {s: metrics(y[systems == s], pred[systems == s]) for s in sorted(set(systems))},
        }
        predictions += [{"test": name, "observation_id": r["observation_id"],
                         "actual_min": float(a), "predicted_min": float(b)}
                        for r, a, b in zip(rows, y, pred)]
        print(name, report["validation"][name]["mae_min"], flush=True)
    model = CatBoostRegressor(**params)
    model.fit(X, np.log1p(y))
    model.save_model(str(args.out / "catboost.cbm"))
    report["training_ranges"] = {k: [min(v), max(v)] for k in numeric
                                 if (v := [number(r["metadata"].get(k)) for r in rows
                                           if number(r["metadata"].get(k)) is not None])}
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2))
    (args.out / "oof_predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    print(json.dumps({"rows": len(rows), "available_rows": available_rows,
                      "systems": len(set(systems)), "output": str(args.out),
                      "validation": {name: {k: v for k, v in result.items() if k not in {"folds", "per_system"}}
                                     for name, result in report["validation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
