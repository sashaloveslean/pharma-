#!/usr/bin/env python3
"""Offline, source-traceable TFV/FTC audit. Never silently approve uncertain labels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from collect_open_methods import license_allowed
from open_retention import load_observations

TARGETS = {"tenofovir": "SGOIRFVFHAKUTI", "emtricitabine": "XQSPYNMVSIKCOC"}
FULL_KEYS = {"tenofovir": "SGOIRFVFHAKUTI-ZCFIWIBFSA-N", "emtricitabine": "XQSPYNMVSIKCOC-NTSWFWBYSA-N"}


def identity_status(compound_id, target):
    if compound_id == FULL_KEYS[target]:
        return "exact_full_inchikey"
    if compound_id.split("-")[0] != TARGETS[target]:
        return "different_connectivity"
    return "connectivity_only_stereochemistry_unresolved_or_different"


def text_of(node):
    return " ".join(" ".join(node.itertext()).split())


def source_number(text):
    return float(text.replace("−", "-").strip())


def read_article(root, pmcid):
    manifest = json.loads((root / "manifest.json").read_text())
    entry = next(r for r in manifest["articles"] if r["pmcid"] == pmcid)
    path = root / (pmcid + ".xml")
    content = path.read_bytes()
    if entry["status"] != "downloaded" or hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"Unverified source {pmcid}")
    article = ET.fromstring(content)
    if not license_allowed(article)[0]:
        raise ValueError(f"Unconfirmed reuse license {pmcid}")
    return article, {"pmcid": pmcid, "source_url": entry["url"], "source_sha256": entry["sha256"],
                     "license_evidence": entry["license_evidence"]}


def retention_conflict(article):
    """Keep both contradictory author claims, rather than selecting the plausible one."""
    paragraph = next(n for n in article.findall(".//body//p")
                     if "retention times for lamivudine" in text_of(n).lower())
    text = text_of(paragraph)
    match = re.search(r"found to be ([\d.,\s]+and\s+[\d.]+) min", text)
    if not match:
        raise ValueError("Unrecognized source retention-time list")
    values = [float(v) for v in re.findall(r"\d+\.\d+", match[1])]
    if len(values) != 9:
        raise ValueError("Unexpected analyte/value count")
    narrative = {"tenofovir": values[4], "emtricitabine": values[2]}
    table = next(t for t in article.findall(".//table-wrap") if t.get("id", "").endswith("-t005"))
    grid = [[text_of(c) for c in row if c.tag in {"td", "th"}] for row in table.findall(".//tr")]
    header = next(row for row in grid if "Retention Time (min)" in row)
    index = header.index("Retention Time (min)")
    tabular = {row[0].lower(): float(row[index]) for row in grid
               if row and row[0].lower() in TARGETS}
    if set(tabular) != set(TARGETS):
        raise ValueError("Target values missing from MS table")
    return {"issue": "contradictory_author_reported_retention_times",
            "narrative_tr_min": narrative, "table_tr_min": tabular,
            "source_table": table.get("id"), "narrative_evidence": text,
            "table_evidence": text_of(table),
            "conflicting_analytes": [name for name in TARGETS if narrative[name] != tabular[name]],
            "action": "quarantine_both_claims_pending_author_or_raw_signal_verification",
            "training_ready": False}


def doe_candidates(article, provenance):
    tables = {t.get("id"): t for t in article.findall(".//table-wrap")}
    grid = [[text_of(c) for c in row if c.tag in {"td", "th"}]
            for row in tables["tab2"].findall(".//tr")]
    # Explicit lookup: published center pH is 3.5, NOT the midpoint 3.25.
    methanol, ph = {-1.0: 10.0, 0.0: 20.0, 1.0: 30.0}, {-1.0: 2.5, 0.0: 3.5, 1.0: 4.0}
    conflicts = [text_of(p) for p in article.findall(".//body//p")
                 if "final optimum conditions" in text_of(p).lower()]
    records = []
    for row in grid:
        if len(row) != 7 or not row[0].isdigit():
            continue
        records.append({**provenance, "record_id": f"PMC3929521:tab2:{row[0]}",
                        "compound": "emtricitabine", "resolution_partner": "lamivudine (internal standard)",
                        "source_table": "tab2", "factor_mapping_table": "tab1", "reported_experiment": int(row[0]),
                        "reported_run_order": int(row[1]), "methanol_percent": methanol[source_number(row[2])],
                        "reported_mobile_phase_pH": ph[source_number(row[3])], "reported_tr_min": float(row[4]),
                        "reported_rs": float(row[5]), "reported_asymmetry": float(row[6]),
                        "column": "Phenomenex C18", "column_length_mm": 250, "column_id_mm": 4.6,
                        "particle_um": 5, "temperature_c": 35, "flow_ml_min": 1.0,
                        "measured_t0_min": None, "peak_width_min": None,
                        "run_id": None, "source_row": row,
                        "issues": ["final_method_pH_3.2_vs_buffer_pH_6.8_in_same_paragraph",
                                   "injection_volume_10_vs_20_ul", "complete_DoE_mobile_phase_recipe_needs_review"],
                        "contradiction_evidence": conflicts, "target_pair": False,
                        "status": "quarantine", "training_ready": False})
    if len(records) != 13:
        raise ValueError("Expected all 13 experimental rows, including center replicates")
    return records


def build(args):
    args.out.mkdir(parents=True, exist_ok=True)
    observations, by_system = [], defaultdict(list)
    for row in load_observations(args.data):
        name = next((name for name, key in TARGETS.items()
                     if row["compound_id"].split("-")[0] == key), None)
        if name is None:
            continue
        identity = identity_status(row["compound_id"], name)
        record = {**row, "target_name": name, "identity_status": identity,
                  "identity_match_scope": "Full InChIKey checked against confirmed TFV/FTC structures",
                  "dataset_role": "target_diagnostic_not_independent_final_test", "training_ready": False,
                  "issues": ["primary_method_and_peak_identity_review_required"] +
                            ([] if identity == "exact_full_inchikey" else ["target_stereochemical_identity_not_confirmed"])}
        observations.append(record)
        by_system[row["system_id"]].append(record)
    pairs = []
    for sid, rows in sorted(by_system.items()):
        grouped = {name: [r["observation_id"] for r in rows if r["target_name"] == name] for name in TARGETS}
        if not all(grouped.values()):
            continue
        pairs.append({"system_id": sid, "observations": grouped, "pair_linkage": "same_method_not_verified_same_injection",
                      "run_id": None, "reported_rs": None, "derived_rs": None,
                      "separation_confirmed": False, "training_ready": False})
    article, provenance = read_article(args.articles, "PMC12114340")
    conflict = {**provenance, **retention_conflict(article)}
    article, provenance = read_article(args.articles, "PMC3929521")
    doe = doe_candidates(article, provenance)
    for filename, records in [("retention_candidates.jsonl", observations), ("doe_quarantine.jsonl", doe)]:
        (args.out / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (args.out / "pair_candidates.json").write_text(json.dumps(pairs, indent=2))
    (args.out / "source_conflicts.json").write_text(json.dumps([conflict], indent=2, ensure_ascii=False))
    with (args.out / "review_queue.csv").open("w", newline="") as f:
        fields = ["record_id", "compound", "source", "status", "issues"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in observations:
            writer.writerow(dict(record_id=r["observation_id"], compound=r["target_name"], source=r["source_url"],
                                 status="review_required", issues="; ".join(r["issues"])))
        for r in doe:
            writer.writerow(dict(record_id=r["record_id"], compound=r["compound"], source=r["source_url"],
                                 status=r["status"], issues="; ".join(r["issues"])))
        for name in conflict["conflicting_analytes"]:
            writer.writerow(dict(record_id=f"PMC12114340:rt_conflict:{name}", compound=name,
                                 source=conflict["source_url"], status="quarantine",
                                 issues=conflict["issue"]))
    summary = {"report_target_observations": len(observations), "by_analyte": dict(Counter(r["target_name"] for r in observations)),
               "identity_status_counts": dict(Counter(r["identity_status"] for r in observations)),
               "same_method_pair_candidates": len(pairs), "verified_same_injection_pairs": 0,
               "verified_target_pair_rs_labels": 0, "doe_rows_quarantined": len(doe),
               "target_article_with_conflicting_rt": conflict, "new_training_approved": 0,
               "dataset_role": "curation_and_development_not_untouched_final_validation"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    report = f"""# Проверка открытых данных: тенофовир + эмтрицитабин

Тенофовир здесь — TFV, не TDF/TAF. Обучение модели и предыдущие результаты не изменены.

- RepoRT: {len(observations)} наблюдений целевых веществ, {len(pairs)} систем с обоими веществами.
- Полный InChIKey совпадает в {sum(r['identity_status'] == 'exact_full_inchikey' for r in observations)} строках;
  остальные имеют совпадение только связности и не подтверждают нужную стереохимию TFV/FTC.
- Общая система не означает общую инъекцию: проверенных пар с ширинами пиков/Rs сейчас 0 в этом аудите.
- Из таблицы PMC3929521 восстановлены все 13 опытов изменения метанола и pH, включая повторные центральные точки.
  Это FTC + ламивудин, НЕ TFV + FTC. Строки помещены в карантин из-за противоречий описания условий.
- Новых автоматически одобренных строк обучения: 0. Это ограничение качества, а не отсутствие найденных чисел.

## Обнаруженное противоречие целевой статьи

PMC12114340, текст: TFV {conflict['narrative_tr_min']['tenofovir']} мин, FTC {conflict['narrative_tr_min']['emtricitabine']} мин.
Таблица 5: TFV {conflict['table_tr_min']['tenofovir']} мин, FTC {conflict['table_tr_min']['emtricitabine']} мин.
Противоречат не только времена, но и порядок выхода. Ни один вариант не выбран автоматически.
Источник и дословный контекст сохранены в source_conflicts.json вместе с SHA256.

## Что нужно проверить дальше

1. Сверить противоречивые времена с исходной хроматограммой/допматериалами; не усреднять их.
2. Для RepoRT подтвердить точную колонку, составы A/B, полный градиент, pH и идентичность пиков.
3. Получить ширины или авторский Rs именно TFV/FTC в одной инъекции. Разницу двух tR не называть Rs.
4. После проверки добавить подтверждённые данные в обучение с разделением по источникам и молекулам.
   Эти примеры уже использованы в разработке; финальную проверку провести на других независимых данных.

review_queue.csv — очередь проверки; retention_candidates.jsonl — данные с условиями и происхождением;
doe_quarantine.jsonl — отдельные опыты, не разрешённые для обучения.
"""
    (args.out / "REPORT.md").write_text(report)
    print(json.dumps({k: v for k, v in summary.items() if k != "target_article_with_conflicting_rt"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/public_retention/report/observations.jsonl"))
    p.add_argument("--articles", type=Path, default=Path("data/external/open_methods"))
    p.add_argument("--out", type=Path, default=Path("reports/target_data_curation"))
    build(p.parse_args())
