#!/usr/bin/env python3
"""Train a provisional CatBoost model for direct retention-time prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from features import compute_descriptors, mol_from_smiles


NUMERIC_CONDITIONS = (
    "column_length_mm", "column_id_mm", "particle_um", "organic_percent",
    "pH", "flow_ml_min", "temperature_c", "buffer", "condition_completeness",
)
CATEGORICAL_CONDITIONS = ("column_chemistry", "modifier")
MOLECULE_DESCRIPTORS = (
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "RingCount", "FractionCSP3",
    "HeavyAtomCount", "FormalCharge", "CarboxylicAcidCount", "AmineCount",
    "AmideCount", "PhenolCount", "SulfonamideCount", "PhosphateCount",
    "HeteroAromaticCount", "HalogenCount",
)
FP_BITS = 128


def optional_float(value) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def featurize(row: dict) -> list:
    from rdkit.Chem import rdFingerprintGenerator

    descriptors = compute_descriptors(row["smiles"])
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FP_BITS
    ).GetFingerprint(mol_from_smiles(row["smiles"]))
    values = [optional_float(descriptors[name]) for name in MOLECULE_DESCRIPTORS]
    values.extend(int(fingerprint.GetBit(bit)) for bit in range(FP_BITS))
    for name in NUMERIC_CONDITIONS:
        if name == "buffer":
            values.append(1.0 if str(row.get(name)).lower() in {"true", "1", "yes"} else 0.0)
        else:
            values.append(optional_float(row.get(name)))
    values.extend((row.get(name) or "__missing__") for name in CATEGORICAL_CONDITIONS)
    return values


def feature_names() -> list[str]:
    return [*MOLECULE_DESCRIPTORS, *[f"morgan_{i}" for i in range(FP_BITS)], *NUMERIC_CONDITIONS, *CATEGORICAL_CONDITIONS]


def main() -> None:
    try:
        from catboost import CatBoostRegressor
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_absolute_error
    except ImportError as exc:
        raise SystemExit("Install dependencies with: .venv/bin/pip install catboost scikit-learn") from exc

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/tr_training.csv")
    parser.add_argument("--model-out", default="models/retention_time_catboost.cbm")
    parser.add_argument("--metadata-out", default="models/retention_time_catboost.json")
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    groups = np.asarray([row["inn"] for row in rows])
    unique_groups = len(set(groups))
    if len(rows) < 10 or unique_groups < 3:
        raise SystemExit(f"Need >=10 rows and >=3 molecules; have {len(rows)} rows/{unique_groups} molecules")

    names = feature_names()
    X = np.asarray([featurize(row) for row in rows], dtype=object)
    y = np.log1p(np.asarray([float(row["retention_time_min"]) for row in rows]))
    weights = np.asarray([float(row["sample_weight"]) for row in rows])
    cat_indices = [names.index(name) for name in CATEGORICAL_CONDITIONS]
    monotone_constraints = {
        names.index("organic_percent"): -1,
        names.index("flow_ml_min"): -1,
        names.index("column_length_mm"): 1,
        names.index("column_id_mm"): 1,
    }
    params = dict(
        iterations=args.iterations, depth=5, learning_rate=0.04,
        loss_function="RMSE", random_seed=42, verbose=False,
        l2_leaf_reg=8, allow_writing_files=False,
        monotone_constraints=monotone_constraints,
    )

    folds = min(5, unique_groups)
    splitter = GroupKFold(n_splits=folds)
    predicted_log = np.empty_like(y)
    for train_indices, test_indices in splitter.split(X, y, groups):
        cv_model = CatBoostRegressor(**params, cat_features=cat_indices)
        cv_model.fit(X[train_indices], y[train_indices], sample_weight=weights[train_indices])
        predicted_log[test_indices] = cv_model.predict(X[test_indices])
    predicted = np.expm1(predicted_log)
    actual = np.expm1(y)
    mae = float(mean_absolute_error(actual, predicted, sample_weight=weights))
    baseline = np.full_like(actual, np.median(actual))
    baseline_mae = float(mean_absolute_error(actual, baseline, sample_weight=weights))

    model = CatBoostRegressor(**params, cat_features=cat_indices)
    model.fit(X, y, sample_weight=weights)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model_out)
    metadata = {
        "model_type": "CatBoostRegressor",
        "target": "log1p(retention_time_min)",
        "prediction_output": "retention_time_min",
        "rows": len(rows), "unique_molecules": unique_groups,
        "group_cv_folds": folds, "group_cv_mae_min": round(mae, 4),
        "median_baseline_mae_min": round(baseline_mae, 4),
        "feature_names": names, "categorical_feature_indices": cat_indices,
        "monotone_constraints": {
            "organic_percent": -1,
            "flow_ml_min": -1,
            "column_length_mm": 1,
            "column_id_mm": 1,
        },
        "limitations": [
            "provisional model trained on automatically extracted observations",
            "many HPLC condition fields are missing",
            "retention time is instrument- and column-geometry-dependent",
            "approximate document values receive reduced sample weight",
        ],
    }
    Path(args.metadata_out).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
