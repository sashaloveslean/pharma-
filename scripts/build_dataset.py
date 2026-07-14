#!/usr/bin/env python3
"""Consolidate per-molecule chromatography labels and join molecular features.

Pipeline position:
    dissolution_chunks.jsonl
      -> extract_conditions.py   (fragment-level condition blocks)
      -> build_dataset.py        (this: one labelled row per molecule + features)
      -> train_model.py

Outputs:
    data/processed/labels.csv    human-readable consolidated conditions per document
    data/processed/dataset.json  modelling table: features (X) + targets (y)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from features import DESCRIPTOR_NAMES, compute_descriptors, morgan_fingerprint


# Target fields the model will learn to predict.
NUMERIC_TARGETS = [
    "column_length_mm",
    "column_id_mm",
    "particle_um",
    "column_temp_c",
    "flow_ml_min",
    "injection_ul",
    "mobile_phase_ph",
    "primary_wavelength_nm",
]
CATEGORICAL_TARGETS = [
    "column_phase",
    "detector",
    "primary_organic",
]
# free-text / boolean conditions carried alongside the method (not scored numerically)
EXTRA_TARGETS = [
    "reagents",
    "mobile_phase",
    "has_buffer",
]


CORE_FIELDS = (
    "column_phase", "column_length_mm", "column_id_mm", "particle_um",
    "column_temp_c", "detector", "wavelengths_nm", "flow_ml_min",
    "injection_ul", "mobile_phase_ph", "mobile_phase_raw",
)


def _completeness(record: dict) -> int:
    return sum(1 for field in CORE_FIELDS if record.get(field) not in (None, [], ""))


def _fill_from(records: list[dict], field: str):
    """First non-empty value for a field, scanning the most complete blocks first."""
    for record in records:
        value = record.get(field)
        if value not in (None, [], ""):
            return value
    return None


def consolidate(records: list[dict]) -> dict:
    """Pick the single most complete real method as this molecule's conditions.

    Blending across blocks produced non-physical values (e.g. 4.33 mm ID) and lost
    the mobile-phase recipe. Instead we transfer one coherent, real method: the
    most complete condition block, back-filling any missing field from the other
    blocks of the same document (ordered by completeness).
    """
    ordered = sorted(records, key=_completeness, reverse=True)
    best = ordered[0]

    wavelengths = _fill_from(ordered, "wavelengths_nm") or []
    primary_wavelength = min(wavelengths) if wavelengths else None

    # reagents are additive across the document's blocks: build one deduped
    # inventory of every named chemical the method needs
    reagents: list[dict] = []
    seen_reagents: set[str] = set()
    for record in ordered:
        for reagent in record.get("reagents") or []:
            if reagent["name"] not in seen_reagents:
                seen_reagents.add(reagent["name"])
                reagents.append(reagent)

    solvents: list[str] = []
    for record in ordered:
        solvents.extend(record.get("solvents") or [])
    solvent_set = set(solvents)
    if "acetonitrile" in solvent_set:
        primary_organic = "acetonitrile"
    elif "methanol" in solvent_set:
        primary_organic = "methanol"
    else:
        primary_organic = None

    return {
        "n_blocks": len(records),
        "column_length_mm": _fill_from(ordered, "column_length_mm"),
        "column_id_mm": _fill_from(ordered, "column_id_mm"),
        "particle_um": _fill_from(ordered, "particle_um"),
        "column_temp_c": _fill_from(ordered, "column_temp_c"),
        "flow_ml_min": _fill_from(ordered, "flow_ml_min"),
        "injection_ul": _fill_from(ordered, "injection_ul"),
        "mobile_phase_ph": _fill_from(ordered, "mobile_phase_ph"),
        "mobile_phase": _fill_from(ordered, "mobile_phase_raw"),
        "primary_wavelength_nm": primary_wavelength,
        "column_phase": _fill_from(ordered, "column_phase"),
        "detector": _fill_from(ordered, "detector"),
        "primary_organic": primary_organic,
        "reagents": reagents,
        "has_buffer": bool({"phosphate_buffer", "acetate_buffer"} & solvent_set),
        "source_page": best.get("start_page"),
    }


def load_molecule_map(path: Path) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            mapping[row["source_file"]] = row
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="data/processed/chromatography_conditions.jsonl")
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--labels-out", default="data/processed/labels.csv")
    parser.add_argument("--dataset-out", default="data/processed/dataset.json")
    args = parser.parse_args()

    # group fragment records by document
    by_source: dict[str, list[dict]] = {}
    with open(args.conditions, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            by_source.setdefault(record["source"], []).append(record)

    molecule_map = load_molecule_map(Path(args.molecule_map))

    labels_rows: list[dict] = []
    dataset_rows: list[dict] = []
    featurized = 0

    for source, records in sorted(by_source.items()):
        consolidated = consolidate(records)
        mapping = molecule_map.get(source, {})
        label_row = {
            "source_file": source,
            "trade_name": mapping.get("trade_name", ""),
            "inn": mapping.get("inn", ""),
            "smiles": mapping.get("smiles", ""),
            **consolidated,
        }
        labels_rows.append(label_row)

        smiles = (mapping.get("smiles") or "").strip()
        if not smiles:
            continue
        try:
            descriptors = compute_descriptors(smiles)
            fingerprint = morgan_fingerprint(smiles)
        except (ValueError, RuntimeError) as exc:
            print(f"  skip {source}: {exc}")
            continue

        dataset_rows.append(
            {
                "source_file": source,
                "inn": mapping.get("inn", ""),
                "smiles": smiles,
                "descriptors": descriptors,
                "fingerprint": fingerprint,
                "targets": {
                    key: consolidated.get(key)
                    for key in NUMERIC_TARGETS + CATEGORICAL_TARGETS + EXTRA_TARGETS
                },
            }
        )
        featurized += 1

    # write human-readable labels
    labels_path = Path(args.labels_out)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(labels_rows[0].keys()) if labels_rows else []
    with labels_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels_rows)

    # write modelling dataset
    dataset_path = Path(args.dataset_out)
    with dataset_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "descriptor_names": list(DESCRIPTOR_NAMES),
                "numeric_targets": NUMERIC_TARGETS,
                "categorical_targets": CATEGORICAL_TARGETS,
                "extra_targets": EXTRA_TARGETS,
                "rows": dataset_rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Documents with conditions : {len(labels_rows)}")
    print(f"Rows with SMILES features : {featurized}")
    print(f"Labels  -> {args.labels_out}")
    print(f"Dataset -> {args.dataset_out}")


if __name__ == "__main__":
    main()
