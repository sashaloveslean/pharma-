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
    "mobile_phase_ph": "pH подвижной фазы",
    "flow_ml_min": "Скорость потока, мл/мин",
    "injection_ul": "Объём ввода, мкл",
}
ORDER = list(LABELS.keys())

REAGENT_CATEGORIES = {
    "organic_solvent": "Органические растворители",
    "buffer_salt": "Соли буфера",
    "acid": "Кислоты (регулировка pH)",
    "base": "Основания (регулировка pH)",
    "modifier": "Модификаторы",
    "ion_pairing": "Ион-парные агенты",
}


ELUTION_LABELS = {"isocratic": "изократический", "gradient": "градиентный"}


def print_mobile_phase(components, ratio, raw, elution_mode, gradient_steps) -> None:
    mode = ELUTION_LABELS.get(elution_mode)
    header = "Подвижная фаза (растворы и соотношение)"
    if mode:
        header += f" — режим: {mode}"
    print(header + ":")

    if components:
        solutions = " : ".join(c["solution"] for c in components)
        parts = " : ".join(str(c["part"]) for c in components)
        print(f"  {solutions}")
        print(f"  соотношение {parts}")
    elif ratio:
        print(f"  соотношение {ratio} (растворы не распознаны)")
    elif raw:
        snippet = raw if len(raw) <= 100 else raw[:100].rstrip() + "…"
        print(f"  {snippet}")
    else:
        print("  — (состав не распознан)")

    if gradient_steps:
        if all("time_min" in step for step in gradient_steps):
            profile = ", ".join(
                f"{step['time_min']} мин → {step['percent_b']}% B" for step in gradient_steps
            )
        else:
            profile = " → ".join(f"{step['percent_b']}% B" for step in gradient_steps)
        print(f"  профиль градиента: {profile}")


def print_reagents(reagents, source, top) -> None:
    print("Прогнозируемые реагенты:")
    if not reagents:
        print("  — (в методе ближайшего аналога реагенты не распознаны)")
        return
    note = ""
    if top and source and source != (top["inn"] or top["source_file"]):
        note = f"  (из {source})"
    by_category: dict[str, list[str]] = {}
    for reagent in reagents:
        by_category.setdefault(reagent["category"], []).append(reagent["name"])
    for category, title in REAGENT_CATEGORIES.items():
        names = by_category.get(category)
        if names:
            print(f"  {title}: {', '.join(names)}")
    if note:
        print(f" {note}")


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
    field_source = result.get("field_source", {})
    analogs = result["analogs"]
    top = analogs[0] if analogs else None

    print(f"SMILES: {smiles}\n")

    print_reagents(prediction.get("reagents"), field_source.get("reagents"), top)

    print()
    print_mobile_phase(
        prediction.get("mobile_phase_components"),
        prediction.get("mobile_phase_ratio"),
        prediction.get("mobile_phase"),
        prediction.get("elution_mode"),
        prediction.get("gradient_steps"),
    )

    print("\nПрогнозируемые условия хроматографирования:")
    for key in ORDER:
        value = prediction.get(key)
        origin = field_source.get(key)
        note = ""
        if top and origin and origin != (top["inn"] or top["source_file"]):
            note = f"  (из {origin})"  # back-filled from a further analog
        print(f"  {LABELS[key]:26s}: {value if value not in (None, '') else '—'}{note}")

    if prediction.get("has_buffer"):
        print("  (подвижная фаза содержит буфер)")

    print("\nМетод перенесён с ближайшего структурного аналога:")
    for i, analog in enumerate(analogs):
        mark = "→" if i == 0 else " "
        print(f"  {mark} {analog['similarity']:.3f}  {analog['inn']}  [{analog['source_file']}]")

    if top and top["similarity"] < 0.3:
        print(
            "\n⚠ Низкое сходство (<0.30): близкого аналога в базе нет, "
            "прогноз ненадёжен — нужна ручная разработка метода."
        )
    print(
        "\nЭто стартовые условия по аналогии с реальным методом из НД, "
        "а не готовая валидированная методика."
    )


if __name__ == "__main__":
    main()
