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

from compound_enrichment import enrich_name, load_cache
from model import NearestAnalogPredictor
from smiles_lookup import fetch_pubchem_smiles

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
MISSING_LABELS = {
    **LABELS,
    "reagents": "Реагенты",
    "elution_mode": "Режим элюирования",
    "gradient_steps": "Градиент",
    "mobile_phase_components": "Компоненты подвижной фазы",
    "mobile_phase_ratio": "Соотношение подвижной фазы",
    "mobile_phase": "Подвижная фаза",
    "has_buffer": "Наличие буфера",
}

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


def print_reagents(reagents) -> None:
    print("Прогнозируемые реагенты:")
    if not reagents:
        print("  — (в базовом методе реагенты не распознаны; не заимствуются из других методик)")
        return
    by_category: dict[str, list[str]] = {}
    for reagent in reagents:
        by_category.setdefault(reagent["category"], []).append(reagent["name"])
    for category, title in REAGENT_CATEGORIES.items():
        names = by_category.get(category)
        if names:
            print(f"  {title}: {', '.join(names)}")


def resolve_smiles(args, molecule_map_path: Path) -> str:
    if args.smiles:
        return args.smiles
    with molecule_map_path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["inn"].strip().lower() == args.inn.strip().lower() and row["smiles"].strip():
                return row["smiles"].strip()
    if not args.lookup_pubchem:
        raise SystemExit(f"No SMILES found for INN {args.inn!r} in {molecule_map_path}")

    result = fetch_pubchem_smiles(args.inn)
    smiles = result["isomeric_smiles"] or result["canonical_smiles"]
    if not smiles:
        raise SystemExit(f"PubChem did not return SMILES for {args.inn!r}")

    print(f"Resolved {args.inn!r} through PubChem: {smiles}\n")
    return smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles")
    group.add_argument("--inn", "--name", dest="inn", help="Look up SMILES locally, then PubChem")
    parser.add_argument("--model", default="models/nearest_analog.json")
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--compound-properties", default="data/processed/compound_properties.json")
    parser.add_argument("--enrich-online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lookup-pubchem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    smiles = resolve_smiles(args, Path(args.molecule_map))
    external_properties = resolve_external_properties(args)
    model = NearestAnalogPredictor.load(args.model)
    result = model.predict_smiles(smiles, external_properties=external_properties)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    prediction = result["prediction"]
    analogs = result["analogs"]
    top = analogs[0] if analogs else None

    print(f"SMILES: {smiles}\n")
    print_external_properties(external_properties)
    print_charge_profile(result.get("query_profile", {}))

    if top:
        print("Базовый метод перенесён целиком с ближайшего структурно-физхимического аналога:")
        print(
            f"  {top['similarity']:.3f}  {top['inn']}  [{top['source_file']}] "
            f"(fp={top.get('fingerprint_similarity')}, desc={top.get('descriptor_similarity')}, "
            f"groups={top.get('functional_group_similarity')}, "
            f"scaffold={top.get('scaffold_similarity')}, ion={top.get('ionization_class')})"
        )
        print()

    print_confidence(result.get("confidence", {}))

    print()
    print_reagents(prediction.get("reagents"))

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
        print(f"  {LABELS[key]:26s}: {value if value not in (None, '') else '—'}")

    if prediction.get("has_buffer"):
        print("  (подвижная фаза содержит буфер)")

    print_missing_fields(result.get("missing_fields", []))
    print_sanity_checks(prediction, result.get("query_profile", {}), top)
    print_screening_plan(prediction, result.get("query_profile", {}))

    print("\nБлижайшие аналоги:")
    for i, analog in enumerate(analogs):
        mark = "→" if i == 0 else " "
        print(
            f"  {mark} {analog['similarity']:.3f}  {analog['inn']}  [{analog['source_file']}] "
            f"(fp={analog.get('fingerprint_similarity')}, desc={analog.get('descriptor_similarity')}, "
            f"groups={analog.get('functional_group_similarity')}, "
            f"scaffold={analog.get('scaffold_similarity')}, ion={analog.get('ionization_class')})"
        )

    if top and top["similarity"] < 0.3:
        print(
            "\n⚠ Низкое сходство (<0.30): близкого аналога в базе нет, "
            "прогноз ненадёжен — нужна ручная разработка метода."
        )
    print(
        "\nЭто стартовые условия по аналогии с реальным методом из НД, "
        "а не готовая валидированная методика."
    )


def print_confidence(confidence: dict) -> None:
    if not confidence:
        return
    print("Оценка применимости:")
    print(f"  Analog confidence     : {confidence.get('analog_confidence')}")
    print(f"  Chemical compatibility: {confidence.get('chemical_compatibility')}")
    print(f"  Field completeness    : {confidence.get('field_completeness')}")
    print(f"  Critical completeness : {confidence.get('critical_completeness')}")
    print(f"  Execution readiness   : {confidence.get('execution_readiness')}")
    print(f"  Structure confidence : {confidence.get('structure_confidence')}")
    print(f"  Reference database   : {confidence.get('reference_database_size')} molecules")
    local = confidence.get("local_database_coverage") or {}
    print(f"  Local coverage       : fp>0.3: {local.get('close_fp_gt_0_3')}, fp>0.2: {local.get('medium_fp_gt_0_2')}")
    print(
        f"  Retrieval stability  : {confidence.get('retrieval_stability')} "
        f"({confidence.get('top1_stability')})"
    )
    print(f"  Neighbor separation  : {confidence.get('neighbor_separation')}")
    print(f"  Overall uncertainty  : {confidence.get('model_uncertainty')}")


def resolve_external_properties(args) -> dict:
    if not args.inn:
        return {}
    cache = load_cache(Path(args.compound_properties))
    key = args.inn.strip().lower()
    if key in cache:
        return cache[key]
    if not args.enrich_online:
        return {}
    try:
        return enrich_name(args.inn)
    except RuntimeError as exc:
        print(f"External enrichment skipped: {exc}\n")
        return {}


def print_external_properties(properties: dict) -> None:
    if not properties:
        return
    pubchem = properties.get("pubchem") or {}
    chembl = properties.get("chembl") or {}
    print("Внешний профиль вещества:")
    if pubchem:
        useful = [
            ("PubChem InChIKey", pubchem.get("InChIKey")),
            ("MW", pubchem.get("MolecularWeight")),
            ("XLogP", pubchem.get("XLogP")),
            ("TPSA", pubchem.get("TPSA")),
            ("HBD/HBA", format_pair(pubchem.get("HBondDonorCount"), pubchem.get("HBondAcceptorCount"))),
        ]
        print("  " + "; ".join(f"{name}: {value}" for name, value in useful if value not in (None, "")))
    if chembl:
        useful = [
            ("ChEMBL", chembl.get("molecule_chembl_id")),
            ("parent", chembl.get("parent_chembl_id")),
            ("alogp", chembl.get("alogp")),
            ("psa", chembl.get("psa")),
            ("acidic pKa (calc.)", chembl.get("cx_most_apka")),
            ("basic pKa (calc.)", chembl.get("cx_most_bpka")),
        ]
        print("  " + "; ".join(f"{name}: {value}" for name, value in useful if value not in (None, "")))
    print()


def print_charge_profile(profile: dict) -> None:
    charge_profile = profile.get("charge_profile")
    if not charge_profile:
        return
    values = ", ".join(
        f"pH {ph}: {'/'.join(f'{charge:+d}' for charge in charges)}"
        for ph, charges in charge_profile.items()
    )
    print(f"pH-aware профиль заряда ({profile.get('charge_profile_source')}):")
    print(f"  {values}")
    print("  Возможные состояния, не количественные доли микроформ.\n")


def format_pair(left, right) -> str | None:
    if left in (None, "") and right in (None, ""):
        return None
    return f"{left}/{right}"


def print_missing_fields(fields: list[str]) -> None:
    if not fields:
        return
    labels = [MISSING_LABELS.get(field, field) for field in fields]
    print("\nНе распознано в базовом методе и не заимствуется из других методик:")
    for label in labels:
        print(f"  - {label}")


def print_sanity_checks(prediction: dict, profile: dict, top: dict | None) -> None:
    warnings = []
    if prediction.get("has_buffer") and not prediction.get("mobile_phase_ph"):
        warnings.append("буфер указан, но pH не распознан — pH нужно задать экспериментально")
        if charge_profile_changes(profile.get("charge_profile")):
            warnings.append(
                "ионное состояние аналита зависит от pH; удерживание может существенно измениться"
            )
    if prediction.get("mobile_phase_ratio") and not prediction.get("mobile_phase_components"):
        warnings.append("есть соотношение фаз, но названия растворов распознаны неполно")
    if prediction.get("column_id_mm") and prediction.get("flow_ml_min"):
        try:
            linear_flow = float(prediction["flow_ml_min"]) / (float(prediction["column_id_mm"]) ** 2)
            if linear_flow > 0.25:
                warnings.append("проверьте линейную скорость: поток может быть высоким для указанного ID колонки")
        except (TypeError, ValueError):
            pass

    mobile_complete = bool(
        prediction.get("mobile_phase_components") and prediction.get("mobile_phase_ratio")
    )
    buffer_complete = not prediction.get("has_buffer") or prediction.get("mobile_phase_ph") is not None
    method_complete = mobile_complete and buffer_complete
    print("\nChemical consistency: " + ("PASS" if not warnings else "WARNING"))
    print("Method completeness: " + ("PASS" if method_complete else "FAIL"))
    if not mobile_complete:
        print("\nПричина:")
        print("  подвижная фаза и её состав не распознаны.")
    elif not buffer_complete:
        print("\nПричина:")
        print("  для буферной подвижной фазы не распознан pH.")
    print("\nОбоснование:")
    if top:
        print(f"  - ближайший аналог: {top.get('inn') or top.get('source_file')}")
        if top.get("functional_group_similarity", 0) >= 0.8:
            print("  - совпадают ключевые функциональные группы")
        if top.get("ionization_similarity", 0) >= 0.8:
            print("  - совпадает pH-зависимый профиль ионизации")
    if mobile_complete:
        print("  - подвижная фаза распознана")
    if not buffer_complete:
        print("  - pH отсутствует, поэтому метод пока невоспроизводим")
    for warning in warnings:
        print(f"  ⚠ {warning}")


def print_screening_plan(prediction: dict, profile: dict) -> None:
    print("\nРекомендуемый screening вместо одной фиксированной точки:")
    mobile_ready = bool(
        prediction.get("mobile_phase_components") and prediction.get("mobile_phase_ratio")
    )
    buffer_ready = not prediction.get("has_buffer") or prediction.get("mobile_phase_ph") is not None
    ratio = parse_ratio(prediction.get("mobile_phase_ratio"))
    if ratio:
        organic = organic_percentage(prediction, ratio)
        low = max(5, organic - 10)
        high = min(95, organic + 10)
        points = sorted({low, organic, high})
        organic_label = {"acetonitrile": "ACN", "methanol": "MeOH"}.get(
            prediction.get("primary_organic"), "Органическая фаза"
        )
        print(f"  {organic_label}: {', '.join(f'{point:g}%' for point in points)}")
    elif prediction.get("primary_organic"):
        print(f"  Органическая фаза ({prediction['primary_organic']}): начать с 20-40% и уточнить удерживание")
    else:
        print("  Органическая фаза: диапазон не задан, требуется ручной подбор")

    if prediction.get("mobile_phase_ph"):
        ph = float(prediction["mobile_phase_ph"])
        print(f"  pH: {max(1.5, ph - 0.5):.1f}-{min(10.0, ph + 0.5):.1f}")
    elif prediction.get("has_buffer"):
        candidates = ph_screening_candidates(profile.get("charge_profile"))
        if candidates:
            print("  pH: " + ", ".join(f"{value:.1f}" for value in candidates))
        else:
            print("  pH: буфер есть, но pH не распознан; задать pH до эксперимента")
    else:
        candidates = ph_screening_candidates(profile.get("charge_profile"))
        if candidates:
            print("  pH: " + ", ".join(f"{value:.1f}" for value in candidates))
        else:
            print("  pH: проверить кислый/нейтральный screening в зависимости от pKa аналита")

    flow = prediction.get("flow_ml_min")
    if flow:
        flow = float(flow)
        if mobile_ready and buffer_ready:
            print(f"  Перенесённый поток: {flow} мл/мин")
            print(f"  Screening потока: {max(0.1, flow * 0.8):.2f}-{flow * 1.2:.2f} мл/мин")
        else:
            print(f"  Поток: пока оставить {flow} мл/мин")

    temp = prediction.get("column_temp_c")
    if temp:
        temp = float(temp)
        if mobile_ready and buffer_ready:
            print(f"  Температура: {max(20, temp - 5):.0f}-{temp + 5:.0f} °C")
        else:
            print(f"  Температура: пока оставить {temp:g} °C")
    else:
        print("  Температура: начать с 25-35 °C")

    candidates = ph_screening_candidates(profile.get("charge_profile"))
    if ratio and candidates:
        organic = organic_percentage(prediction, ratio)
        organic_points = sorted({max(5, organic - 10), organic, min(95, organic + 10)})
        combinations = rank_screening_combinations(organic_points, organic, candidates)
        print("  Приоритетные комбинации:")
        for rank, (organic_value, ph) in enumerate(combinations, 1):
            print(f"    #{rank} ACN {organic_value:g}%, pH {ph:.1f}")


def parse_ratio(value) -> list[float] | None:
    if not value:
        return None
    parts = [part.strip() for part in str(value).replace("/", ":").split(":")]
    try:
        numbers = [float(part.replace(",", ".")) for part in parts if part]
    except ValueError:
        return None
    return numbers if len(numbers) >= 2 else None


def charge_profile_changes(profile: dict | None) -> bool:
    if not profile:
        return False
    ordered = [tuple(profile[key]) for key in sorted(profile, key=float)]
    return any(left != right for left, right in zip(ordered, ordered[1:]))


def ph_screening_candidates(profile: dict | None) -> list[float]:
    """Place discrete screening points across the pH region where charge changes."""
    if not profile:
        return []
    points = sorted((float(ph), tuple(charges)) for ph, charges in profile.items())
    changing = [
        (left[0], right[0])
        for left, right in zip(points, points[1:])
        if left[1] != right[1]
    ]
    if not changing:
        return []
    # Concentrate measurements in the widest observed transition interval;
    # sparse profile points imply the greatest uncertainty there.
    start, end = max(changing, key=lambda interval: interval[1] - interval[0])
    candidates = []
    step = 0.5 if end - start >= 1.5 else max(0.25, (end - start) / 3)
    value = start
    while value < end and len(candidates) < 4:
        candidates.append(round(value, 2))
        value += step
    return candidates or [round((start + end) / 2.0, 1)]


def organic_percentage(prediction: dict, ratio: list[float]) -> float:
    components = prediction.get("mobile_phase_components") or []
    organic_names = ("ацетонитрил", "acetonitrile", "метанол", "methanol")
    for component in components:
        if any(name in str(component.get("solution", "")).lower() for name in organic_names):
            total = sum(ratio)
            part = float(component.get("part", 0))
            return 100.0 * part / total if total else part
    return min(ratio)


def rank_screening_combinations(
    organic_points: list[float],
    transferred_organic: float,
    ph_points: list[float],
) -> list[tuple[float, float]]:
    """Rule-based top experiments: baseline first, then composition and pH probes."""
    base = min(organic_points, key=lambda value: abs(value - transferred_organic))
    high = max(organic_points)
    low_ph, high_ph = min(ph_points), max(ph_points)
    ranked = [(base, low_ph), (high, low_ph), (base, high_ph)]
    result: list[tuple[float, float]] = []
    for item in ranked:
        if item not in result:
            result.append(item)
    return result


if __name__ == "__main__":
    main()
