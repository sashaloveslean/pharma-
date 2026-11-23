#!/usr/bin/env python3
"""Audit flow and cycle time; generate volume-preserving transfers, not claimed optima."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from method_physics import linear_velocity_mm_s, bed_volume_ml, gradient_segments

FLOW_SOURCE = "https://assets.thermofisher.com/TFS-Assets/CMD/manuals/Man-065184-Acclaim-RSLC-Columns-Man065184-EN.pdf"
VOLUME_SOURCE = "https://support.waters.com/KB_Chem/Columns/WKB51034_What_is_the_suggested_column_volumes_needed_to_clean_a_column_during_a_gradient_run"


def flow_transfer(program, flow, length, diameter):
    original = float(program[0]["flow rate [ml/min]"])
    if any(float(p["flow rate [ml/min]"]) != original for p in program):
        raise ValueError("This transfer supports constant-flow sources only")
    new = deepcopy(program)
    ratio = original/flow
    for p in new:
        p["t [min]"] = float(p["t [min]"])*ratio
        p["flow rate [ml/min]"] = flow
    # Generic screening policy in EMPTY geometric volumes, not pore volumes.
    # Unknown dwell volume is not silently assigned zero; times remain lower bounds.
    geometric = bed_volume_ml(length, diameter)
    segments = gradient_segments(new)
    for role, minimum_cv in [("high_organic_wash",3), ("re_equilibration",5)]:
        relevant = [(i,s) for i,s in enumerate(segments) if s["role"] == role]
        if len(relevant) != 1:
            raise ValueError("Expected one explicit wash and re-equilibration hold")
        i,s = relevant[0]
        extra = max(0,minimum_cv*geometric-s["volume_ml"])/flow
        for p in new[i+1:]:
            p["t [min]"] += extra
        segments = gradient_segments(new)
    return new


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    source = json.loads(args.method.read_text())
    method = source["selected"]
    meta,program = method["conditions"],method["gradient_program"]
    diameter,length = float(meta["column.id"]),float(meta["column.length"])
    geometric = bed_volume_ml(length,diameter)
    segments = gradient_segments(program)
    last = max(method["predicted_tr_min"].values())
    upper = max(v[1] for v in method["predicted_intervals_90_nominal_min"].values()) if "predicted_intervals_90_nominal_min" in method else None
    audit = {"origin": "flow/gradient copied from public method, not outputs of CatBoost",
             "empty_geometric_volume_ml": geometric, "segments": segments,
             "superficial_u_mm_s": linear_velocity_mm_s(float(meta["column.flowrate"]),diameter),
             "pore_velocity_mm_s": None, "porosity": None,
             "predicted_last_target_peak_min": last, "last_peak_interval_upper_min": upper,
             "typical_flow_ml_min": [0.3,0.8], "typical_is_not_optimal": True,
             "van_deemter_optimum": None,"source_flow_below_typical": float(meta["column.flowrate"])<0.3,
             "sources": [FLOW_SOURCE,VOLUME_SOURCE],"flow_candidates": []}
    for s in segments:
        s["empty_column_volumes"] = s["volume_ml"]/geometric
        if upper is not None and s["role"] == "analytical_ramp_or_initial_hold":
            s["time_after_target_interval_upper_min"] = max(0,s["end_min"]-max(s["start_min"],upper))
    for flow in (0.2,0.35,0.4,0.5):
        candidate = flow_transfer(program,flow,length,diameter)
        item = {"flow_ml_min":flow,"superficial_u_mm_s":linear_velocity_mm_s(flow,diameter),
                "pore_velocity_mm_s": None,"gradient_program":candidate,
                "cycle_time_without_unknown_dwell_compensation_min":candidate[-1]["t [min]"],
                "pressure_status":"unknown; measured pressure and instrument limit required",
                "predicted_tr_min":None,"predicted_Rs":None,"execution_ready":False,
                "role":"volume-preserving screening proposal, not a Van Deemter optimum",
                "policy":"preserve source ramp/wash volumes; minimum wash 3 and re-equilibration 5 empty CV",
                "missing_dwell_compensation":True}
        audit["flow_candidates"].append(item)
        with (args.out/f"flow_{flow:g}_gradient.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(candidate[0]))
            writer.writeheader(); writer.writerows(candidate)
    (args.out/"audit.json").write_text(json.dumps(audit,indent=2))
    wash = next(s for s in segments if s["role"]=="high_organic_wash")
    eq = next(s for s in segments if s["role"]=="re_equilibration")
    text = ("# Поток и длительность метода: проверка\n\n"
            "Поток и длинный градиент в предыдущем кандидате перенесены из публикации; "
            "CatBoost прогнозировал tR, а не выбирал их как оптимальные.\n\n"
            f"При 0,2 мл/мин и ID 2,1 мм поверхностная скорость — {audit['superficial_u_mm_s']:.3f} мм/с. "
            "Скорость с учётом пористости не рассчитана: пористость неизвестна. "
            "В формуле нужны согласованные единицы, а смысл ε должен быть задан (общая или межчастичная пористость).\n\n"
            "Типичный диапазон производителя — 0,3–0,8 мл/мин. Это не строгое окно оптимума "
            "Ван-Деемтера. Для оптимума нужны характеристики эффективности и условия переноса массы; "
            "универсальный штраф только по диаметру и размеру частиц не обоснован.\n\n"
            f"Промывка 17–25 мин: {wash['volume_ml']:.2f} мл, {wash['empty_column_volumes']:.2f} геометрических CV. "
            "Это согласуется с общим ориентиром 3–5 CV промывки. "
            f"Возврат к исходному составу 25,1–30 мин даёт {eq['empty_column_volumes']:.2f} CV; "
            "это ниже общего ориентира 5–10 CV уравновешивания и требует проверки воспроизводимости.\n\n"
            "## Скрининг потока\n\n| F, мл/мин | u поверхностная, мм/с | Нижняя оценка цикла, мин |\n|---:|---:|---:|\n")
    for c in audit["flow_candidates"]:
        text += f"| {c['flow_ml_min']:g} | {c['superficial_u_mm_s']:.3f} | {c['cycle_time_without_unknown_dwell_compensation_min']:.2f} |\n"
    text += ("\nПрограммы сохраняют объёмы исходных градиента и промывки. "
             "Уравновешивание доведено как минимум до 5 пустых геометрических CV — явная "
             "скрининговая политика, а не гарантия для этой системы. Неизвестный dwell volume "
             "потребует дополнительного времени. Давление, новые tR и Rs не выдумываются.\n\n"
             "Оценка затрат цикла и растворителя должна использоваться вместе с достоверным Rs/неопределённостью. "
             "Автоматическое отбрасывание промывки по одному прогнозируемому tR не реализовано.\n\n"
             f"Источники: {FLOW_SOURCE}\n\n{VOLUME_SOURCE}\n")
    (args.out/"REPORT.md").write_text(text)
    print(text)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method",type=Path,default=Path("reports/tenofovir_method_calibrated/method_candidates.json"))
    p.add_argument("--out",type=Path,default=Path("reports/method_efficiency_audit"))
    main(p.parse_args())
