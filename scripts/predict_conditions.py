#!/usr/bin/env python3
"""Predict HPLC conditions for a new molecule from its SMILES.

Example:
    python3 scripts/predict_conditions.py --smiles "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
    python3 scripts/predict_conditions.py --inn ibuprofen
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from model import NearestAnalogPredictor

LABELS = {
    "column_phase": "Стационарная фаза",
    "column_length_mm": "Длина колонки, мм",
    "column_id_mm": "Внутр. диаметр, мм",
    "particle_um": "Размер частиц, мкм",
    "column_temp_c": "Температура колонки, °C",
    "detector": "Детектор",
    "primary_wavelength_nm": "Длина волны, нм",
    "primary_organic": "Органический модификатор",
    "flow_ml_min": "Скорость потока, мл/мин",
    "injection_ul": "Объём ввода, мкл",
    "mobile_phase_ph": "pH подвижной фазы",
}
ORDER = list(LABELS.keys())


def resolve_smiles(args, molecule_map_path: Path) -> str:
    if args.smiles:
        return args.smiles
    with molecule_map_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["inn"].strip().lower() == args.inn.strip().lower() and row["smiles"].strip():
                return row["smiles"].strip()
    raise SystemExit(f"No SMILES found for INN {args.inn!r} in {molecule_map_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles")
    group.add_argument("--inn", help="Look up SMILES for this INN in the molecule map")
    parser.add_argument("--model", default="models/nearest_analog.json")
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    smiles = resolve_smiles(args, Path(args.molecule_map))
    model = NearestAnalogPredictor.load(args.model)
    result = model.predict_smiles(smiles)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prediction = result["prediction"]
    print(f"SMILES: {smiles}\n")
    print("Прогноз условий хроматографа:")
    for key in ORDER:
        value = prediction.get(key)
        print(f"  {LABELS[key]:28s}: {value if value is not None else '—'}")

    print("\nПо аналогии с (структурное сходство Танимото):")
    for analog in result["analogs"]:
        print(f"  {analog['similarity']:.3f}  {analog['inn']}  [{analog['source_file']}]")
    print(
        "\nПрогноз — стартовая точка метода по ближайшим аналогам, "
        "а не готовая валидированная методика."
    )


if __name__ == "__main__":
    main()
