#!/usr/bin/env python3
"""Compare baseline/chemistry models with disjoint selection, calibration and tests."""
import argparse
from collections import Counter
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from demo_tenofovir import eligible, TARGETS
from open_retention import load_observations
from realistic_features import ChargeCache, VERSION, HYDRO_VERSION, enriched_vector
from train_open_retention import vector


def split_groups(groups, seed=20261023, fractions=(0.6, 0.1, 0.15)):
    unique = np.asarray(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(unique)
    ends = [int(len(unique) * sum(fractions[:i])) for i in (1, 2, 3)]
    if min(ends[0], ends[1]-ends[0], ends[2]-ends[1], len(unique)-ends[2]) < 1:
        raise ValueError("Insufficient independent groups")
    sets = np.split(unique, ends)
    return [np.flatnonzero(np.isin(groups, values)) for values in sets]


def metric(y, pred, groups):
    errors = np.abs(y-pred)
    return {"mae_min": float(np.mean(errors)), "median_absolute_error_min": float(np.median(errors)),
            "macro_group_mae_min": float(np.mean([np.mean(errors[groups == g]) for g in set(groups)]))}


def group_radius(y, pred, groups, alpha=0.1):
    scores = sorted(max(np.abs(y[groups == g]-pred[groups == g])) for g in set(groups))
    rank = math.ceil((len(scores)+1)*(1-alpha))
    return scores[rank-1] if rank <= len(scores) else float("inf")


def evaluate_intervals(y, pred, groups, radius):
    covered = np.abs(y-pred) <= radius
    return {"radius_min": float(radius), "row_coverage": float(np.mean(covered)),
            "all_rows_in_group_coverage": float(np.mean([np.all(covered[groups == g]) for g in set(groups)])),
            "mean_width_min": float(np.mean(pred+radius-np.maximum(0, pred-radius))),
            "scope": "Empirical held-out coverage; exchangeable-group assumption may fail across sources"}


def provenance(row):
    info = row["source_info"][0]
    if "smrt" in str(info.get("name", "")).lower() or "Domingo-Almenara" in str(info.get("source", "")):
        return "METLIN_SMRT"
    return info.get("url", "").strip().lower().rstrip("/") or row["system_id"]


def predictions(model, X):
    return np.maximum(0, np.expm1(model.predict(X)))


def fit(X, y, train, iterations, weights=None):
    model = CatBoostRegressor(iterations=iterations, depth=5, learning_rate=0.06,
                              random_seed=42, verbose=False, allow_writing_files=False,
                              thread_count=4, cat_features=[X.shape[1]-2, X.shape[1]-1])
    model.fit(X[train], np.log1p(y[train]), sample_weight=weights[train] if weights is not None else None)
    return model


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    demo = json.loads((args.demo / "result.json").read_text())
    all_rows = load_observations(args.data)
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != demo["dataset_sha256"]:
        raise ValueError("Dataset differs from the recorded demo dataset")
    by_id = {r["observation_id"]: r for r in all_rows}
    rows = [by_id[i] for i in demo["training_observation_ids"]]
    assert all(eligible(r) for r in rows)
    numeric = demo["numeric_metadata_features"]
    # Freeze the original holdouts before adding any new training observations.
    original_groups = np.asarray([r["compound_id"].split("-")[0] for r in rows])
    train, validation, calibration, test = split_groups(original_groups)
    original_sources = np.asarray([provenance(r) for r in rows])
    source_train, source_selection, source_cal, source_test = split_groups(original_sources, fractions=(0.65, 0.05, 0.15))
    heldout_molecules = set(original_groups[np.concatenate([validation, calibration, test])])
    heldout_sources = set(original_sources[np.concatenate([source_selection, source_cal, source_test])])
    original_count = len(rows)
    if args.extra_training_rows:
        existing = set(demo["training_observation_ids"])
        candidates = [r for r in all_rows if r["observation_id"] not in existing and eligible(r)
                      and r["source_info"] and r["source_info"][0].get("method.type") == "RP"
                      and r["compound_id"].split("-")[0] not in heldout_molecules]
        # Round-robin systems to avoid spending all additional rows on SMRT.
        from collections import defaultdict
        pools = defaultdict(list)
        for r in candidates:
            pools[r["system_id"]].append(r)
        rng = np.random.default_rng(20261023)
        for pool in pools.values():
            rng.shuffle(pool)
        extra = []
        while pools and len(extra) < args.extra_training_rows:
            for sid in sorted(list(pools)):
                extra.append(pools[sid].pop())
                if not pools[sid]:
                    del pools[sid]
                if len(extra) == args.extra_training_rows:
                    break
        rows.extend(extra)
        train = np.concatenate([train, np.arange(original_count, len(rows))])
        source_train = np.concatenate([source_train, np.asarray([
            i for i in range(original_count, len(rows)) if provenance(rows[i]) not in heldout_sources], dtype=int)])
    cache = ChargeCache(args.cache)
    arrays = {"baseline": [], "chemistry_ph": []}
    for i, row in enumerate(rows, 1):
        arrays["baseline"].append(vector(row, numeric))
        arrays["chemistry_ph"].append(enriched_vector(row, numeric, cache, args.feature_version))
        if i % 250 == 0:
            cache.save()
            print(f"Featurized {i}/{len(rows)}; charge cache={len(cache.records)}", flush=True)
    cache.save()
    X = {k: np.asarray(v, dtype=object) for k, v in arrays.items()}
    y = np.asarray([r["retention_time_min"] for r in rows])
    groups = np.asarray([r["compound_id"].split("-")[0] for r in rows])
    weights = None
    if args.domain_weighting:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        anchors = [generator.GetFingerprint(Chem.MolFromSmiles(info["payload"]["PropertyTable"]["Properties"][0]["SMILES"]))
                   for info in demo["pubchem"].values()]
        fingerprint_cache = {}
        similarities = []
        for row in rows:
            structure = row["smiles"]
            if structure not in fingerprint_cache:
                fingerprint_cache[structure] = generator.GetFingerprint(Chem.MolFromSmiles(structure))
            similarities.append(max(DataStructs.TanimotoSimilarity(fingerprint_cache[structure], a) for a in anchors))
        # Uses target STRUCTURES only, never target retention labels or target test errors.
        weights = 1 + 9 * np.asarray(similarities)**2
    assert not set(groups[train]) & set(groups[np.concatenate([validation, calibration, test])])
    report = {"feature_version": args.feature_version, "seed": 20261023, "rows": len(rows),
              "iterations": args.iterations, "dataset_sha256": demo["dataset_sha256"],
              "additional_training_rows": len(rows)-original_count,
              "additional_training_sampling": "round-robin systems, frozen holdout molecules excluded",
              "training_weights": "1 + 9 * max_Tanimoto_to_TFV_FTC_squared; structure only" if weights is not None else "uniform",
              "systems_sha256": hashlib.sha256((args.data.parent / "systems.json").read_bytes()).hexdigest(),
              "numeric_metadata_features": numeric,
              "excluded_target_family_keys": demo["excluded_connectivity_keys"],
              "split_sizes": dict(zip(("train", "selection", "calibration", "test"), map(len, (train, validation, calibration, test)))),
              "selection_rule": "minimum molecule-macro MAE on separate selection set; test/TFV/FTC not used",
              "models": {}, "charge_cache_status": dict(Counter(r["status"] for r in cache.records.values())),
              "limitations": ["Subsample experiment, not full-corpus training",
                              "Charge enumeration yields possibilities, not quantitative species fractions or measured pKa",
                              "TFV/FTC motivated development: target results are diagnostic, not untouched final validation",
                              "No peak width or Rs model; no validated separation recommendation",
                              "Source provenance and metadata can contain unresolved duplicates/inconsistencies"]}
    models = {}
    for name, features in X.items():
        models[name] = fit(features, y, train, args.iterations, weights)
        models[name].save_model(str(args.out / (name + ".cbm")))
        report["models"][name] = {
            "feature_count": features.shape[1],
            "selection": metric(y[validation], predictions(models[name], features[validation]), groups[validation])}
    selected = min(models, key=lambda name: report["models"][name]["selection"]["macro_group_mae_min"])
    report["selected_model"] = selected
    for name, features in X.items():
        pred = predictions(models[name], features[test])
        q = group_radius(y[calibration], predictions(models[name], features[calibration]), groups[calibration])
        report["models"][name]["test"] = metric(y[test], pred, groups[test])
        report["models"][name]["intervals_90_nominal"] = evaluate_intervals(y[test], pred, groups[test], q)
        print(name, report["models"][name], flush=True)
    source_groups = np.asarray([provenance(r) for r in rows])
    assert not set(source_groups[source_train]) & set(source_groups[np.concatenate([source_cal, source_test])])
    source_model = fit(X[selected], y, source_train, args.iterations, weights)
    source_pred = predictions(source_model, X[selected][source_test])
    source_q = group_radius(y[source_cal], predictions(source_model, X[selected][source_cal]), source_groups[source_cal])
    report["new_source_test"] = {
        **metric(y[source_test], source_pred, source_groups[source_test]),
        "test_rows": len(source_test), "test_source_groups": len(set(source_groups[source_test])),
        "training_rows": len(source_train),
        "intervals_90_nominal": evaluate_intervals(y[source_test], source_pred, source_groups[source_test], source_q)}
    targets = [r for r in all_rows if r["compound_id"].split("-")[0] in TARGETS.values()
               and r["source_info"][0].get("method.type") == "RP"]
    report["target_diagnostic"] = {}
    for name, model in models.items():
        feature_rows = [vector(r, numeric) if name == "baseline" else enriched_vector(r, numeric, cache, args.feature_version) for r in targets]
        pred = predictions(model, np.asarray(feature_rows, dtype=object))
        q = report["models"][name]["intervals_90_nominal"]["radius_min"]
        records = [{"observation_id": r["observation_id"], "name": r["name"], "system_id": r["system_id"],
                    "measured_min": r["retention_time_min"], "predicted_min": float(p),
                    "interval_90_min": [max(0, float(p-q)), float(p+q)]} for r, p in zip(targets, pred)]
        report["target_diagnostic"][name] = {
            "mae_min": float(np.mean([abs(r["measured_min"]-r["predicted_min"]) for r in records])), "records": records}
    cache.save()
    report["split_observation_ids"] = {name: [rows[i]["observation_id"] for i in indices]
                                       for name, indices in zip(("train", "selection", "calibration", "test"),
                                                                 (train, validation, calibration, test))}
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2))
    write_report(args.out, report)
    print(json.dumps({"selected_model": selected, "models": report["models"],
                      "new_source_test": report["new_source_test"],
                      "target_mae": {k: v["mae_min"] for k, v in report["target_diagnostic"].items()}}, indent=2))


def write_report(out, r):
    text = "# Проверка приближения модели к эксперименту\n\n"
    text += ("Сравнены прежние и расширенные признаки на одинаковых наблюдениях и разбиениях. "
             "Выбор модели, калибровка интервала и тест используют разные молекулы. "
             "TFV/FTC и TDF/TAF исключены из всех этих частей.\n\n")
    text += "| Признаки | Тест MAE, мин | Macro MAE, мин | Покрытие групп интервалом |\n|---|---:|---:|---:|\n"
    for name, m in r["models"].items():
        text += (f"| {name} | {m['test']['mae_min']:.3f} | {m['test']['macro_group_mae_min']:.3f} | "
                 f"{100*m['intervals_90_nominal']['all_rows_in_group_coverage']:.1f}% |\n")
    text += f"\nПо отдельной выборке выбора выбран **{r['selected_model']}**.\n\n"
    text += ("Расширение: fingerprint 1024 бит, функциональные группы (включая фосфоновые кислоты), "
             "перечень возможных зарядов при pH метода из Dimorphite-DL, взаимодействия pH/групп, "
             "геометрия колонки и действительный состав смеси с учётом предварительно смешанных A/B. "
             "Перечень зарядов не является распределением долей форм и не заменяет экспериментальный pKa.\n\n")
    if r["feature_version"] == HYDRO_VERSION:
        text += ("Версия v2 дополнительно использует поверхностную линейную скорость, скорость с пористостью "
                 "только если пористость известна, максимальную положительную крутизну %B/мин "
                 "и длительности аналитической части, промывки и уравновешивания. "
                 "Это признаки регрессии удерживания, не модель эффективности Ван-Деемтера.\n\n")
    selected = r["models"][r["selected_model"]]
    q = selected["intervals_90_nominal"]["radius_min"]
    text += (f"Для выбранной модели радиус калиброванного интервала: {q:.2f} мин. "
             "Нижняя граница обрезается на нуле. Номинальные 90% относятся к принятому "
             "предположению обменности групп; эмпирическое покрытие показано выше. "
             "Оно не гарантируется на новом приборе или вне химической области данных.\n\n")
    text += (f"Тест новых источников: MAE {r['new_source_test']['mae_min']:.2f} мин, "
             f"{r['new_source_test']['test_rows']} строк. Покрытие групп — "
             f"{100*r['new_source_test']['intervals_90_nominal']['all_rows_in_group_coverage']:.1f}%, "
             "что ниже номинальных 90%. Перенос интервалов на новые системы не подтверждён.\n\n")
    text += "## Диагностика TFV/FTC\n\n| Модель | MAE, мин |\n|---|---:|\n"
    for name, diagnostic in r["target_diagnostic"].items():
        text += f"| {name} | {diagnostic['mae_min']:.3f} |\n"
    text += ("\nЭтот пример уже использовался для диагностики при разработке, поэтому он не считается "
             "нетронутым финальным тестом. Истинный финальный тест нужно зафиксировать отдельно.\n\n"
             "## Граница готовности\n\n"
             "Прогноз tR и неопределённости не подтверждает разделение смеси. Ширины пиков, Rs, "
             "давление, матрица и перенос на прибор заказчика пока не валидированы. "
             "При широких пересекающихся интервалах ранжирование условий должно оставаться нерешённым.\n")
    (out / "REPORT.md").write_text(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/public_retention/report/observations.jsonl"))
    p.add_argument("--demo", type=Path, default=Path("reports/tenofovir_demo"))
    p.add_argument("--cache", type=Path, default=Path("data/processed/public_retention/research_cache/charges.json"))
    p.add_argument("--out", type=Path, default=Path("reports/realism_evaluation"))
    p.add_argument("--iterations", type=int, default=250)
    p.add_argument("--extra-training-rows", type=int, default=0,
                   help="Add system-balanced training rows without altering the original holdouts")
    p.add_argument("--domain-weighting", action="store_true", help="Upweight chemical similarity to target structures, never target RT labels")
    p.add_argument("--feature-version", choices=[VERSION, HYDRO_VERSION], default=VERSION)
    main(p.parse_args())
