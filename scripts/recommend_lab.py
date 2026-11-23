#!/usr/bin/env python3
"""Evidence-first lab review packet. A first experiment is not a validated optimum."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import re

from propose_pair_method import SPEC


PREFLIGHT = [
    "Аналитик подтверждает TFV (не TDF/TAF), FTC, чистоту стандартов и состав матрицы.",
    "Подтвердить точный артикул/геометрию колонки и паспортные пределы pH, растворителей, температуры и давления.",
    "Проверить совместимость прибора; предел давления задать не выше меньшего предела прибора и колонки.",
    "Уточнить способ задания/измерения pH в источнике и проверить приготовленные фазы; не считать процент кислоты измеренным pH.",
    "Измерить/учесть dwell volume и проверить достаточность уравновешивания повторными инъекциями.",
    "Согласовать предложенные объём ввода, концентрацию, растворитель пробы и UV-детекцию; проверить растворимость и чувствительность.",
]


def cycle_issues(program):
    if len(program) < 2 or float(program[0]["t [min]"]) != 0:
        return ["missing_complete_cycle"]
    initial = [float(program[0][f"{c} [%]"]) for c in "ABCD"]
    final = [float(program[-1][f"{c} [%]"]) for c in "ABCD"]
    if initial != final:
        return ["missing_return_to_initial"]
    before_final = [float(program[-2][f"{c} [%]"]) for c in "ABCD"]
    return [] if before_final == initial else ["missing_explicit_reequilibration_hold"]


def choose_experiments(comparison, validator):
    eligible, rejected = [], []
    for c in comparison["candidates"]:
        system = {"metadata": c["conditions"], "gradient_program": c["gradient_program"], "source_info": c["method_source"]}
        problems = validator(system)
        cycle = cycle_issues(c["gradient_program"]) if not problems else []
        problems.extend(p for p in cycle if p != "missing_explicit_reequilibration_hold")
        if c.get("origin") != "source_program" or c.get("time_scale") != 1:
            problems.append("not_an_unmodified_source_program")
        if not c.get("metadata_source") or not c.get("method_source"):
            problems.append("missing_source_traceability")
        if problems:
            rejected.append({"candidate_id": c["candidate_id"], "reasons": sorted(set(problems))})
        else:
            eligible.append(c)
    def key(c):
        # Lexicographic policy, not learned performance or a misleading confidence score.
        publication = any(info.get("url") and info.get("source", "").startswith("Publication") for info in c["method_source"])
        reviewed_geometry = c.get("manufacturer_limits_checked") and c.get("column") == SPEC["name"]
        # Existing low-pH reference is a screening policy only, not proof of TFV selectivity.
        low_ph_reference = reviewed_geometry and c["source_system"] == "report:0386"
        return (bool(cycle_issues(c["gradient_program"])), -bool(reviewed_geometry), -publication, -bool(low_ph_reference),
                -c.get("same_system_training_rows", 0), c["candidate_id"])
    ranked = sorted(eligible, key=key)
    if ranked and cycle_issues(ranked[0]["gradient_program"]):
        rejected.extend({"candidate_id": c["candidate_id"], "reasons": cycle_issues(c["gradient_program"])} for c in ranked)
        ranked = []  # No primary recipe without an explicit return and equilibration segment.
    selected, used_columns = [], set()
    for c in ranked:
        if c["column"] in used_columns:
            continue
        selected.append(c)
        used_columns.add(c["column"])
        if len(selected) == 3:
            break
    plans = []
    for index, c in enumerate(selected):
        cycle = cycle_issues(c["gradient_program"])
        plans.append({"role": "primary" if index == 0 else "backup", "priority": index+1,
                      "candidate": deepcopy(c), "status": "BLOCKED_REEQUILIBRATION_PROTOCOL" if cycle else "FOR_ANALYST_REVIEW_BEFORE_LAB_TEST",
                      "critical_cycle_issues": cycle,
                      "why_selected": ["Исходная программа без масштабирования времени",
                                       "Пройдена проверка обязательных полей и баланса растворителей",
                                       "Паспортные границы известны для этой геометрии" if c.get("manufacturer_limits_checked") else
                                       "Паспортные границы ещё нужно проверить",
                                       "Приоритет источника и покрытия, а не максимального предсказанного ΔtR"],
                      "column_spec": deepcopy(SPEC) if c["column"] == SPEC["name"] else None,
                      "screening_assumptions_not_source": {"sample": "TFV + FTC reference standards, no matrix assumed",
                                                           "concentration_ug_ml_each": 10, "injection_ul": 1,
                                                           "diluent": "initial mobile phase, only if solubility confirmed",
                                                           "detector": "DAD 200–320 nm; exploratory channels 260 and 280 nm"},
                      "preflight": PREFLIGHT + ([] if c.get("manufacturer_limits_checked") else
                                                ["Блокер варианта: паспорт колонки ещё не проверен."]) +
                                   (["Блокер: источник не задаёт выдержку после возврата к исходному составу. Аналитик должен определить и проверить уравновешивание до повторных инъекций."] if cycle else []),
                      "record_in_lab": ["tR TFV/FTC", "ширины на полувысоте и измеренный Rs", "идентичность/чистота и хвостование пиков",
                                        "давление", "повторяемость", "t0 с указанием маркера и способа измерения, если измерен"],
                      "separation_confirmed": False, "instrument_execution_authorized": False})
    return {"status": "FIRST_EXPERIMENT_PROPOSED" if plans else "BLOCKED_NO_COMPLETE_SOURCE_METHOD",
            "selection_policy": "source completeness, checked manufacturer geometry, published provenance, existing low-pH reference, coverage; ML gap ignored",
            "primary": plans[0] if plans else None, "backups": plans[1:],
            "rejected": rejected, "backup_count": max(0, len(plans)-1),
            "separation_rank_status": comparison.get("rank_status"),
            "instrument_execution_authorized": False,
            "warning": "Первый предлагаемый эксперимент, не лучший доказанный метод. Проверки аналитика обязательны."}


def write_packet(out, recommendation):
    from docx import Document
    doc = Document()
    title = "TFV + FTC: задание на первый лабораторный скрининг"
    doc.add_heading(title, 0)
    intro = "Предлагается для проверки аналитиком перед тестами. Разделение не подтверждено; автоматический запуск не разрешён."
    doc.add_paragraph(intro)
    md = f"# {title}\n\n{intro}\n\n"
    plans = ([recommendation["primary"]] if recommendation["primary"] else []) + recommendation["backups"]
    active = {f"lab_{plan['role']}_{plan['priority']}_gradient.csv" for plan in plans}
    recommendation["active_gradient_exports"] = sorted(active)
    for old in out.iterdir():
        if re.fullmatch(r"lab_(?:primary|backup)_\d+_gradient\.csv", old.name) and old.name not in active:
            archive = out / "superseded_lab_gradients"
            archive.mkdir(exist_ok=True)
            target, version = archive / old.name, 1
            while target.exists():
                target = archive / (old.stem + f"_v{version}.csv"); version += 1
            old.rename(target)
    if not plans:
        message = "Подходящий полный метод не найден. Причины отказа сохранены в lab_recommendation.json; рецепт не достраивается автоматически."
        doc.add_paragraph(message); md += message + "\n"
    for plan in plans:
        c = plan["candidate"]; m = c["conditions"]
        heading = ("Основной вариант" if plan["role"] == "primary" else f"Запасной вариант {plan['priority']-1}") + " — " + c["source_system"]
        if plan["critical_cycle_issues"]:
            heading += " — БЛОКЕР: уравновешивание не задано"
        doc.add_heading(heading, level=1); md += f"## {heading}\n\n"
        def paragraph(text):
            nonlocal md
            doc.add_paragraph(text); md += text + "\n\n"
        paragraph(f"Колонка: {c['column']}; {m['column.length']} × {m['column.id']} мм; частицы {m['column.particle.size']} мкм. "
                  f"Температура: {m['column.temperature']} °C. pH канала A по источнику: {m['eluent.A.pH']} (способ измерения уточнить).")
        if plan["column_spec"]:
            paragraph(f"Артикул: {plan['column_spec']['part_number']}. Руководство: {plan['column_spec']['source_url']}")
        for channel in "ABCD":
            if not any(float(p[f"{channel} [%]"]) > 0 for p in c["gradient_program"]):
                continue
            composition = {k.removeprefix(f"eluent.{channel}."): v for k, v in m.items()
                           if k.startswith(f"eluent.{channel}.") and v not in ("", "0", None)}
            paragraph(f"Фаза {channel}, значения источника: " + json.dumps(composition, ensure_ascii=False))
        if plan["column_spec"]:
            paragraph("Паспортные границы: pH 2–8, температура до 60 °C, вода до 90%, поток до 1,2 мл/мин; "
                      "предел давления для этой геометрии 11000 psi. Поток 0,2 мл/мин перенесён из источника, "
                      "ниже типичного диапазона 0,3–0,8 мл/мин; это не вычисленный оптимум эффективности.")
        paragraph("Почему выбран: " + "; ".join(plan["why_selected"]) + ".")
        paragraph("Предложено для согласования, не извлечено из источника: отдельные стандарты и смесь без матрицы; "
                  "по 10 мкг/мл, ввод 1 мкл, растворитель — исходная подвижная фаза при подтверждённой растворимости. "
                  "DAD 200–320 нм, обзорные каналы 260/280 нм. Последовательность: холостой раствор → отдельные стандарты → смесь, два повтора. "
                  "Это начальный скрининг, не валидация.")
        headers = ["Время, мин", "A, %", "B, %", "C, %", "D, %", "Поток, мл/мин"]
        table = doc.add_table(rows=1, cols=6); table.style = "Light Shading Accent 1"
        for cell, h in zip(table.rows[0].cells, headers): cell.text = h
        md += "| " + " | ".join(headers) + " |\n|" + "---|"*6 + "\n"
        for point in c["gradient_program"]:
            values = [point[k] for k in ["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
            for cell, v in zip(table.add_row().cells, values): cell.text = str(v)
            md += "| " + " | ".join(map(str, values)) + " |\n"
        md += "\n"
        paragraph("Промывка и уравновешивание сохранены из источника; их достаточность проверяется по стабильности давления, "
                  "базовой линии и повторяемости tR. Не сокращать программу только по прогнозу последнего пика.")
        prediction = c.get("predicted_tr_min")
        if prediction:
            paragraph(f"Справочный прогноз модели: TFV {prediction['tenofovir']:.2f} мин; FTC {prediction['emtricitabine']:.2f} мин. "
                      "Не использован для выбора: модель ошибается на целевой паре; порядок пиков и Rs не гарантированы.")
        paragraph("Источник метода: " + "; ".join(info.get("url") or "первичная публикация не указана" for info in c["method_source"]))
        paragraph("Версионированные метаданные: " + c["metadata_source"])
        for check in plan["preflight"]: paragraph("Перед тестом: " + check)
        paragraph("Записать: " + "; ".join(plan["record_in_lab"]) + ". Критерии приемлемости согласует аналитик под задачу; не подменять Rs разницей tR.")
        with (out / f"lab_{plan['role']}_{plan['priority']}_gradient.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(c["gradient_program"][0])); writer.writeheader(); writer.writerows(c["gradient_program"])
    doc.save(out / "Задание_в_лабораторию.docx")
    (out / "LAB_PLAN.md").write_text(md)
    (out / "lab_recommendation.json").write_text(json.dumps(recommendation, ensure_ascii=False, indent=2))


def main():
    from compare_columns import method_issues
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, default=Path("reports/customer_multicolumn/comparison.json"))
    p.add_argument("--out", type=Path, default=Path("reports/customer_multicolumn"))
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    recommendation = choose_experiments(json.loads(args.comparison.read_text()), method_issues)
    write_packet(args.out, recommendation)
    print(json.dumps({"status": recommendation["status"],
                      "primary": recommendation["primary"]["candidate"]["candidate_id"] if recommendation["primary"] else None,
                      "backups": [r["candidate"]["candidate_id"] for r in recommendation["backups"]],
                      "instrument_execution_authorized": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
