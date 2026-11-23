#!/usr/bin/env python3
"""Generate constraint-checked TFV/FTC screening methods on one public column."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from train_open_retention import vector
from realistic_features import ChargeCache, enriched_vector
from method_physics import linear_velocity_mm_s, bed_volume_ml, gradient_segments

MANUAL = "https://assets.thermofisher.com/TFS-Assets/CMD/manuals/Man-065184-Acclaim-RSLC-Columns-Man065184-EN.pdf"
SPEC = {"name": "Thermo Scientific Acclaim RSLC 120 C18", "part_number": "068982",
        "length_mm": 100, "id_mm": 2.1, "particle_um": 2.2, "ph_range": [2, 8],
        "max_temperature_c": 60, "max_water_percent": 90,
        "typical_flow_ml_min": [0.3,0.8],
        "max_flow_ml_min": 1.2, "max_pressure_psi": 11000, "source_url": MANUAL}


def possible_gap(left, right):
    """Propagate individual RT bounds; this is not a calibrated joint Rs interval."""
    minimum = max(0, left[0]-right[1], right[0]-left[1])
    maximum = max(abs(left[1]-right[0]), abs(right[1]-left[0]))
    return [minimum, maximum]


def check_method(meta, gradient):
    problems = []
    if meta.get("column.name") != SPEC["name"]:
        problems.append("wrong_column")
    for key, expected in [("column.length", SPEC["length_mm"]),
                          ("column.id", SPEC["id_mm"]), ("column.particle.size", SPEC["particle_um"])]:
        if float(meta[key]) != expected:
            problems.append("wrong_column_geometry")
    ph = float(meta["eluent.A.pH"])
    if not SPEC["ph_range"][0] <= ph <= SPEC["ph_range"][1]:
        problems.append("pH_outside_column_range")
    if float(meta["column.temperature"]) > SPEC["max_temperature_c"]:
        problems.append("temperature_above_column_limit")
    previous = -1
    for point in gradient:
        t = float(point["t [min]"])
        if t <= previous:
            problems.append("non_increasing_gradient_time")
        previous = t
        fractions = [float(point[f"{channel} [%]"]) for channel in "ABCD"]
        if any(v < 0 or v > 100 for v in fractions) or abs(sum(fractions) - 100) > 1e-6:
            problems.append("gradient_mass_balance")
        water = sum(fraction * float(meta.get(f"eluent.{channel}.h2o") or 0) / 100
                    for fraction, channel in zip(fractions, "ABCD"))
        if water > SPEC["max_water_percent"] + 1e-6:
            problems.append("water_above_column_compatibility")
        if not 0 < float(point["flow rate [ml/min]"]) <= SPEC["max_flow_ml_min"]:
            problems.append("flow_outside_column_range")
    return sorted(set(problems))


def build(args):
    args.out.mkdir(parents=True, exist_ok=True)
    systems = json.loads(args.systems.read_text())
    demo = json.loads((args.demo / "result.json").read_text())
    model = CatBoostRegressor()
    research = json.loads(args.research_evaluation.read_text()) if args.research_evaluation else None
    selected_model = research["selected_model"] if research else "target_held_out"
    model_path = args.research_evaluation.parent / (selected_model + ".cbm") if research else args.demo / "target_held_out.cbm"
    if research and research["dataset_sha256"] != demo["dataset_sha256"]:
        raise ValueError("Research model and demo dataset versions differ")
    model.load_model(str(model_path))
    cache = ChargeCache(args.charge_cache)
    radius = research["models"][selected_model]["intervals_90_nominal"]["radius_min"] if research else None
    smiles = {name: info["payload"]["PropertyTable"]["Properties"][0]["SMILES"]
              for name, info in demo["pubchem"].items()}
    candidates, rejected = [], []
    for sid, system in systems.items():
        meta = system["metadata"]
        if meta.get("column.name") != SPEC["name"] or sid not in {f"report:038{i}" for i in range(2, 7)}:
            continue
        problems = check_method(meta, system["gradient_program"])
        if problems:
            rejected.append({"system_id": sid, "reasons": problems})
            continue
        for scale in (0.8, 1.0, 1.2):
            row = deepcopy(system)
            for point in row["gradient_program"]:
                point["t [min]"] = round(float(point["t [min]"]) * scale, 4)
            problems = check_method(row["metadata"], row["gradient_program"])
            assert not problems
            featurize = (lambda r: enriched_vector(r, demo["numeric_metadata_features"], cache, research["feature_version"])) if selected_model == "chemistry_ph" else (lambda r: vector(r, demo["numeric_metadata_features"]))
            predictions = {name: max(0, float(np.expm1(model.predict([
                featurize({**row, "smiles": structure})])[0])))
                           for name, structure in smiles.items()}
            candidates.append({"candidate_id": f"{sid}:time_x{scale}", "source_system": sid,
                               "conditions": row["metadata"], "gradient_program": row["gradient_program"],
                               "gradient_time_scale": scale, "origin": "public_method" if scale == 1 else "generated_variant",
                               "predicted_tr_min": predictions,
                               "predicted_time_gap_min": abs(predictions["tenofovir"] - predictions["emtricitabine"]),
                               "cycle_time_min": float(row["gradient_program"][-1]["t [min]"]),
                               "constraint_check": "PASS", "predicted_Rs": None,
                               "method_source": row["source_info"],
                               "execution_ready": False})
            segments = gradient_segments(row["gradient_program"])
            geometric = bed_volume_ml(float(meta["column.length"]),float(meta["column.id"]))
            velocities = [linear_velocity_mm_s(float(p["flow rate [ml/min]"]),float(meta["column.id"])) for p in row["gradient_program"]]
            candidates[-1]["physics_diagnostics"] = {
                "superficial_velocity_mm_s_min_max": [min(velocities),max(velocities)],
                "pore_velocity": None, "porosity": None, "van_deemter_optimum": None,
                "below_manufacturer_typical_flow": any(float(p["flow rate [ml/min]"])<0.3 for p in row["gradient_program"]),
                "segments": [{**s,"empty_column_volumes":s["volume_ml"]/geometric} for s in segments],
                "total_solvent_ml": sum(s["volume_ml"] for s in segments),
                "cost_status":"cycle/solvent costs are reported; optimality requires separation and pressure validation"}
            if radius is not None:
                intervals = {name: [max(0, value-radius), value+radius] for name, value in predictions.items()}
                candidates[-1]["predicted_intervals_90_nominal_min"] = intervals
                candidates[-1]["possible_time_gap_min"] = possible_gap(*intervals.values())
    if not candidates:
        raise ValueError("No methods passed column constraints")
    # Time gap is a surrogate, not a separation metric. Do not certify the winner.
    ranked = sorted(candidates, key=lambda r: (-r["predicted_time_gap_min"], r["cycle_time_min"]))
    gap_spread = ranked[0]["predicted_time_gap_min"] - ranked[-1]["predicted_time_gap_min"]
    error_scale = research["target_diagnostic"][selected_model]["mae_min"] if research else demo["benchmark"]["mae_min"]
    resolved = gap_spread > 2 * error_scale
    if research:
        resolved = resolved and any(
            candidate["possible_time_gap_min"][0] > max(other["possible_time_gap_min"][1]
                for other in candidates if other is not candidate) for candidate in candidates)
    # If ML cannot distinguish candidates, show the balanced unmodified low-pH
    # reference as a starting experiment, with the choice explicitly rule-based.
    selected = ranked[0] if resolved else next(r for r in candidates
                                             if r["source_system"] == "report:0386" and r["gradient_time_scale"] == 1)
    result = {"analytes": "Tenofovir TFV + emtricitabine FTC; standard mixture, no matrix",
              "column": SPEC, "rank_status": "numerical_ranking_requires_validation" if resolved else "UNRESOLVED",
              "ranking_surrogate": "predicted absolute tR difference; not Rs",
              "gap_spread_min": gap_spread, "held_out_target_mae_min": error_scale,
              "ranking_gate": "gap spread > 2 * target MAE; heuristic diagnostic, not a statistical interval",
              "selection_basis": "unvalidated model surrogate" if resolved else "rule-based unmodified low-pH reference, not ML optimum",
              "selected": selected, "candidates": ranked, "rejected": rejected,
              "screening_defaults": {"injection_ul": 1, "sample": "mixed TFV/FTC reference standards in initial mobile phase",
                                     "sample_concentration_ug_ml_each": 10,
                                     "detector": "DAD: collect 200–320 nm; exploratory channels 260 and 280 nm",
                                     "replicates": 2, "blank_first": True,
                                     "origin": "proposed screening assumptions; not extracted or validated"},
              "unresolved": ["Instrument maximum pressure, dwell volume and detector configuration unknown",
                             "Measured pressure and peak widths unavailable; cannot predict Rs",
                             "Confirm sample solubility, UV response and peak identity with individual standards",
                             "pH measurement convention in source needs confirmation",
                             "Confirm equilibration by stable pressure/baseline and repeatability; full method is not validated",
                             "Current model has already failed the TFV/FTC elution-order test"],
              "execution_ready": False, "separation_confirmed": False}
    if research:
        result["research_model"] = {"model": selected_model, "path": str(model_path),
                                    "evaluation": str(args.research_evaluation),
                                    "test": research["models"][selected_model]["test"],
                                    "interval_validation": research["models"][selected_model]["intervals_90_nominal"],
                                    "new_source_interval_validation": research["new_source_test"]["intervals_90_nominal"],
                                    "scope": "Measured held-out coverage; not guaranteed for generated gradients/new instruments"}
        result["ranking_gate"] += "; additionally require dominance of propagated RT-interval gap bounds"
        cache.save()
    (args.out / "method_candidates.json").write_text(json.dumps(result, indent=2))
    with (args.out / "gradient.csv").open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=list(selected["gradient_program"][0]))
        w.writeheader()
        w.writerows(selected["gradient_program"])
    report(args.out, result)
    print(json.dumps({"rank_status": result["rank_status"], "selection_basis": result["selection_basis"],
                      "candidates": len(candidates), "rejected": rejected, "gap_spread_min": gap_spread,
                      "selected_id": selected["candidate_id"], "predictions": selected["predicted_tr_min"],
                      "out": str(args.out)}, indent=2))


def report(out, result):
    from docx import Document
    r, spec = result["selected"], result["column"]
    meta = r["conditions"]
    low_ph = r["source_system"] == "report:0386"
    phase_a = "Вода + 0,1% муравьиной кислоты" if low_ph else "Состав A: см. JSON исходного метода"
    phase_b = "Метанол + 0,1% муравьиной кислоты" if low_ph else "Состав B: см. JSON исходного метода"
    title = "TFV + FTC: кандидат полного скринингового метода"
    items = [
        ("Статус", "Исследовательский кандидат, не валидированная методика. "
         f"Рассчитано {len(result['candidates'])} вариантов. Статус ML-ранжирования: {result['rank_status']}. "
         "Разница прогнозов недостаточна для уверенного выбора оптимума. "
         "Выбран исходный низко-pH метод по явному правилу, а не как доказанный лучший вариант."
         if result["rank_status"] == "UNRESOLVED" else "Исследовательский кандидат по суррогатному критерию разницы tR. Разделение не подтверждено."),
        ("Вещества", "Тенофовир TFV (не TDF/TAF) + эмтрицитабин FTC. Предполагается смесь чистых стандартов без матрицы."),
        ("Колонка", f"{spec['name']}, P/N {spec['part_number']}, 100 × 2,1 мм, 2,2 мкм."),
        ("Подвижная фаза A", phase_a), ("Подвижная фаза B", phase_b),
        ("pH", f"Источник сообщает pH A {meta['eluent.A.pH']}. Не считать 0,1% кислоты гарантией точного pH; "
         "уточнить способ измерения и проверить при приготовлении."),
        ("Температура и поток", f"{meta['column.temperature']} °C; поток по таблице. Проценты B обозначают долю канала B, а не всегда общий процент метанола."),
        ("Ввод пробы — предложено", "1 мкл; по 10 мкг/мл каждого стандарта в исходной подвижной фазе, только после проверки растворимости. Это стартовые допущения, не данные источника."),
        ("Детекция — предложено", "DAD 200–320 нм, обзорные каналы 260 и 280 нм. Проверить чувствительность, спектр и специфичность; оптимальные длины волн модель не предсказывает."),
        ("Последовательность — предложено", "Холостой раствор → отдельные стандарты для идентификации пиков → смесь, два повтора. Перед анализом дождаться стабильного давления и базовой линии."),
        ("Прогноз удерживания", f"TFV: {r['predicted_tr_min']['tenofovir']:.2f} мин; FTC: {r['predicted_tr_min']['emtricitabine']:.2f} мин. "
         f"Средняя ошибка целевой диагностики: {result['held_out_target_mae_min']:.2f} мин. "
         "Это не индивидуальный доверительный интервал. Порядок выхода и Rs не гарантированы."),
        ("Границы колонки", "По руководству: pH 2–8, температура до 60 °C, вода до 90%, поток до 1,2 мл/мин. "
         "Для этой геометрии предел давления 11000 psi. Рабочее давление не предсказано; предел на приборе "
         "нужно задать не выше меньшего из допустимых пределов колонки и конкретного прибора."),
        ("Что предсказано и что перенесено", "CatBoost оценивает tR для сгенерированных программ. "
         "Градиентные варианты получены масштабированием времени 0,8/1,0/1,2. "
         "Геометрия, фазы, pH и температура перенесены из открытых методов. "
         "Ввод, проба и детекция — явные скрининговые допущения. Успешное разделение пока не предсказано."),
    ]
    physical = r["physics_diagnostics"]
    items.append(("Гидродинамика и время цикла", f"Поверхностная скорость: {physical['superficial_velocity_mm_s_min_max'][0]:.3f}–"
                  f"{physical['superficial_velocity_mm_s_min_max'][1]:.3f} мм/с. Пористость неизвестна. "
                  "Типичный диапазон производителя: 0,3–0,8 мл/мин; это не вычисленный оптимум Ван-Деемтера. "
                  f"Расход за цикл: {physical['total_solvent_ml']:.2f} мл. "
                  "Разгон градиента, промывка и уравновешивание учитываются отдельно в JSON. "
                  "Промывка не удаляется по одному прогнозируемому времени последнего пика."))
    if "research_model" in result:
        validation = result["research_model"]["interval_validation"]
        ranges = r["predicted_intervals_90_nominal_min"]
        items.append(("Калиброванная неопределённость", f"TFV: {ranges['tenofovir'][0]:.2f}–{ranges['tenofovir'][1]:.2f} мин; "
                      f"FTC: {ranges['emtricitabine'][0]:.2f}–{ranges['emtricitabine'][1]:.2f} мин. "
                      f"Эмпирическое покрытие молекулярных групп: {100*validation['all_rows_in_group_coverage']:.1f}%. "
                      "Интервалы пересекаются: уверенно выбрать порядок выхода и лучший метод нельзя. "
                      "Для сгенерированных программ и новых приборов это покрытие не гарантировано."))
    table = [["Время, мин", "A, %", "B, %", "Поток, мл/мин"]] + [
        [str(p["t [min]"]), str(p["A [%]"]), str(p["B [%]"]), str(p["flow rate [ml/min]"])]
        for p in r["gradient_program"]]
    md = "# " + title + "\n\n" + "\n\n".join(f"## {h}\n\n{v}" for h, v in items)
    md += "\n\n## Программа\n\n| " + " | ".join(table[0]) + " |\n|---|---|---|---|\n"
    md += "\n".join("| " + " | ".join(line) + " |" for line in table[1:])
    md += f"\n\nРуководство колонки: {MANUAL}\n\nМетод-источник: {r['method_source'][0]['url']}\n"
    (out / "METHOD.md").write_text(md)
    doc = Document()
    doc.add_heading(title, 0)
    for heading, text in items:
        doc.add_heading(heading, 2)
        doc.add_paragraph(text)
    doc.add_heading("Программа градиента", 2)
    t = doc.add_table(rows=0, cols=4)
    t.style = "Light Shading Accent 1"
    for line in table:
        for cell, value in zip(t.add_row().cells, line):
            cell.text = value
    doc.add_paragraph("Метод-источник: " + r["method_source"][0]["url"])
    doc.add_paragraph("Руководство колонки: " + MANUAL)
    doc.save(out / "Кандидат_метода_TFV_FTC.docx")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--systems", type=Path, default=Path("data/processed/public_retention/report/systems.json"))
    p.add_argument("--demo", type=Path, default=Path("reports/tenofovir_demo"))
    p.add_argument("--out", type=Path, default=Path("reports/tenofovir_method"))
    p.add_argument("--research-evaluation", type=Path)
    p.add_argument("--charge-cache", type=Path, default=Path("data/processed/public_retention/research_cache/charges.json"))
    build(p.parse_args())
