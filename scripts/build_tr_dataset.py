#!/usr/bin/env python3
"""Build a provisional retention-time dataset that does not require t0."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "observation_id", "inn", "smiles", "column_chemistry", "column_length_mm",
    "column_id_mm", "particle_um", "modifier", "organic_percent", "pH",
    "buffer", "flow_ml_min", "temperature_c", "retention_time_min",
    "sample_weight", "is_approximate", "condition_completeness", "source_file",
    "source_page", "extraction_confidence", "evidence",
]
CONDITION_FIELDS = (
    "column_chemistry", "modifier", "organic_percent", "flow_ml_min", "pH",
    "column_length_mm", "column_id_mm", "particle_um", "temperature_c",
)
APPROXIMATE_RE = re.compile(r"около|примерн|ориентировоч|типич|approximately|about", re.I)
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}


def build_rows(input_path: Path) -> tuple[list[dict], dict]:
    with input_path.open(encoding="utf-8") as file:
        candidates = list(csv.DictReader(file))

    rows, seen = [], set()
    for row in candidates:
        if not row.get("smiles") or not row.get("retention_time_min"):
            continue
        try:
            retention_time = float(row["retention_time_min"])
        except ValueError:
            continue
        if not 0.05 <= retention_time <= 300:
            continue

        # Page-overlap produces duplicate mentions of the same experiment.
        identity = (
            row.get("inn"), row.get("source_file"), retention_time,
            row.get("column_chemistry"), row.get("modifier"),
            row.get("organic_percent"), row.get("pH"), row.get("flow_ml_min"),
            row.get("temperature_c"),
        )
        if identity in seen:
            continue
        seen.add(identity)

        completeness = sum(bool(row.get(field)) for field in CONDITION_FIELDS) / len(CONDITION_FIELDS)
        approximate = bool(APPROXIMATE_RE.search(row.get("evidence") or ""))
        confidence = CONFIDENCE_WEIGHT.get((row.get("extraction_confidence") or "").lower(), 0.4)
        weight = confidence * (0.25 + 0.75 * completeness) * (0.6 if approximate else 1.0)
        built = {field: row.get(field, "") for field in FIELDS}
        built.update({
            "retention_time_min": retention_time,
            "sample_weight": round(weight, 4),
            "is_approximate": int(approximate),
            "condition_completeness": round(completeness, 4),
        })
        rows.append(built)

    report = {
        "candidate_observations": len(candidates),
        "retention_rows_after_deduplication": len(rows),
        "unique_molecules": len({row["inn"] for row in rows}),
        "approximate_rows": sum(row["is_approximate"] for row in rows),
        "rows_with_all_condition_fields": sum(row["condition_completeness"] == 1 for row in rows),
        "target": "retention_time_min",
        "warning": "Provisional tR dataset; tR is system- and geometry-dependent and missing conditions remain informative missing values.",
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/qsrr_observations.csv")
    parser.add_argument("--output", default="data/processed/tr_training.csv")
    parser.add_argument("--report", default="data/processed/tr_report.json")
    args = parser.parse_args()
    rows, report = build_rows(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
