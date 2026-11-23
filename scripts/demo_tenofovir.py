#!/usr/bin/env python3
"""Reproducible target-held-out TFV/FTC demonstration with real reference RTs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from open_retention import api, load_observations, number
from train_open_retention import vector

TARGETS = {"tenofovir": "SGOIRFVFHAKUTI", "emtricitabine": "XQSPYNMVSIKCOC"}
EXCLUDED = {*TARGETS.values(), "JFVZFKDSXNQEJW", "LDEKQSIMHVQZJK"}


def eligible(row):
    return (row["compound_id"].split("-")[0] not in EXCLUDED
            and not any(term in (row.get("name") or "").lower() for term in TARGETS))


def build(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_observations(args.data)
            if r["source_info"] and r["source_info"][0].get("method.type") == "RP"]
    pool = [r for r in rows if eligible(r)]
    indices = sorted(np.random.default_rng(42).choice(len(pool), min(args.train_rows, len(pool)), replace=False))
    train = [pool[i] for i in indices]
    assert all(eligible(r) for r in train)
    numeric = sorted({k for r in train for k, value in r["metadata"].items()
                      if k not in {"id", "column.t0"} and number(value) is not None})
    X = np.asarray([vector(r, numeric) for r in train], dtype=object)
    model = CatBoostRegressor(iterations=args.iterations, depth=5, learning_rate=0.06,
                              random_seed=42, verbose=False, allow_writing_files=False,
                              thread_count=4, cat_features=[X.shape[1]-2, X.shape[1]-1])
    model.fit(X, np.log1p([r["retention_time_min"] for r in train]))
    model.save_model(str(out / "target_held_out.cbm"))
    predictions = {}
    pubchem = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in train]
    for name, prefix in TARGETS.items():
        path = out / f"pubchem_{name}.json"
        if args.refresh_pubchem or not path.exists():
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + quote(name)
                   + "/property/SMILES,InChIKey,MolecularWeight,XLogP,TPSA/JSON")
            payload = api(url)
            path.write_text(json.dumps({"source_url": url, "payload": payload}, indent=2))
        record = json.loads(path.read_text())
        properties = record["payload"]["PropertyTable"]["Properties"]
        if len(properties) != 1 or properties[0]["InChIKey"].split("-")[0] != prefix:
            raise ValueError(f"Identity mismatch for {name}")
        pubchem[name] = record
        mol = Chem.MolFromSmiles(properties[0]["SMILES"])
        similarities = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), fps)
        matches = [r for r in rows if r["compound_id"].split("-")[0] == prefix]
        for row in matches:
            estimate = max(0, float(np.expm1(model.predict([vector(row, numeric)])[0])))
            sid = row["system_id"]
            local = [sim for sim, training_row in zip(similarities, train) if training_row["system_id"] == sid]
            predictions[row["observation_id"]] = {
                "name": name, "system_id": sid, "observation_id": row["observation_id"],
                "measured_tr_min": row["retention_time_min"], "predicted_tr_min": estimate,
                "absolute_error_min": abs(estimate - row["retention_time_min"]),
                "training_rows_same_system": len(local),
                "max_local_tanimoto_2048bit": max(local) if local else None,
                "source_url": row["source_url"],
            }
    pairs = []
    for sid in sorted({r["system_id"] for r in predictions.values()}):
        reference = next(r for r in rows if r["system_id"] == sid)
        analytes = {name: [p for p in predictions.values() if p["system_id"] == sid and p["name"] == name]
                    for name in TARGETS}
        if not all(analytes.values()):
            continue
        unambiguous = all(len(v) == 1 for v in analytes.values())
        pair = {"system_id": sid, "analytes": analytes,
                "conditions": reference["metadata"], "gradient_program": reference["gradient_program"],
                "source_info": reference["source_info"], "unambiguous_reference": unambiguous,
                "predicted_Rs": None, "separation_confirmed": False}
        if unambiguous:
            a, b = [analytes[name][0] for name in TARGETS]
            observed, predicted = b["measured_tr_min"] - a["measured_tr_min"], b["predicted_tr_min"] - a["predicted_tr_min"]
            pair.update(measured_delta_tr_min=observed, predicted_delta_tr_min=predicted,
                        delta_absolute_error_min=abs(observed - predicted),
                        elution_order_correct=bool(observed * predicted > 0))
        pairs.append(pair)
    # Predefined demonstration method: complete public pH metadata, one RT per analyte.
    # All other matching systems are retained in JSON, including ambiguous records.
    selected = next(p for p in pairs if p["system_id"] == "report:0382")
    measured = [p for p in predictions.values()]
    result = {
        "status": "target_held_out_retention_benchmark_not_validated_separation",
        "analytes": "Tenofovir (TFV, not TDF/TAF) + emtricitabine (FTC)",
        "pubchem": pubchem, "training_rows": len(train), "rp_rows_available": len(rows),
        "excluded_target_family_rows": len(rows) - len(pool),
        "excluded_connectivity_keys": sorted(EXCLUDED), "seed": 42, "iterations": args.iterations,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "numeric_metadata_features": numeric,
        "training_observation_ids": [r["observation_id"] for r in train],
        "benchmark": {"target_observations": len(measured),
                      "mae_min": float(np.mean([p["absolute_error_min"] for p in measured])),
                      "by_analyte": {name: {"observations": sum(p["name"] == name for p in measured),
                           "mae_min": float(np.mean([p["absolute_error_min"] for p in measured if p["name"] == name]))}
                                     for name in TARGETS}},
        "target_predictions": measured,
        "pairs": pairs, "selected_system": selected["system_id"],
        "limitations": ["No co-injection/run identity or peak widths: Rs and mixture separation are unverified",
                        "Known-system test on a fixed training subsample; not validation on a new customer system",
                        "Reference metadata and peak identities require verification against primary data",
                        "Selected method uses a variable-flow gradient, not a fixed-flow isocratic method",
                        "No calibrated per-molecule prediction interval"]}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    with (out / "benchmark.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    write_report(out, result, selected)
    print(json.dumps({"training_rows": len(train), "excluded_target_family_rows": len(rows)-len(pool),
                      "benchmark": result["benchmark"], "selected_system": selected["system_id"],
                      "selected_predictions": [selected["analytes"][name][0] for name in TARGETS],
                      "elution_order_correct": selected["elution_order_correct"],
                      "report": str(out / "Отчёт_тенофовир_эмтрицитабин.docx")}, indent=2))


def write_report(out, result, pair):
    a, b = [pair["analytes"][name][0] for name in TARGETS]
    meta = pair["conditions"]
    title = "Тенофовир + эмтрицитабин: демонстрация прототипа"
    intro = ("Использованы только открытые экспериментальные данные. Проверяется тенофовир (TFV), "
             "а не его пролекарства TDF/TAF. Данные заказчика не использованы.")
    method = (f"Открытая система {pair['system_id']}: {meta['column.name']}, "
              f"{meta['column.length']} × {meta['column.id']} мм, {meta['column.particle.size']} мкм; "
              f"температура {meta['column.temperature']} °C; pH {meta['eluent.A.pH']}. "
              "A: вода/метанол 90:10; B: метанол. В обоих каналах: формиат аммония 5 мМ "
              "и муравьиная кислота 0,01%. Поток меняется по программе ниже.")
    protocol = (f"CatBoost обучен на {result['training_rows']} других наблюдениях. "
                f"Все {result['excluded_target_family_rows']} найденных записей TFV, FTC, TDF и TAF "
                "(по connectivity InChIKey и названиям) исключены до выбора обучающей подвыборки. "
                "Измеренные времена целевых веществ не использованы "
                "для обучения; они открыты только для сравнения. Это проверка новых веществ "
                "в известных системах, а не перенос на прибор заказчика.")
    comparison = [["Вещество", "Измерено, мин", "Прогноз, мин", "Ошибка, мин"]] + [
        ["Тенофовир" if r["name"] == "tenofovir" else "Эмтрицитабин",
         f"{r['measured_tr_min']:.2f}", f"{r['predicted_tr_min']:.2f}", f"{r['absolute_error_min']:.2f}"] for r in [a, b]]
    assessment = (f"Разность времён FTC − TFV: измерено {pair['measured_delta_tr_min']:.2f} мин, "
                  f"прогноз {pair['predicted_delta_tr_min']:.2f} мин. "
                  f"Порядок выхода {'совпал' if pair['elution_order_correct'] else 'не совпал'}. "
                  "Разность времён не равна разрешению Rs. Без ширин пиков и подтверждения одного "
                  "запуска нельзя утверждать, что смесь разделена.")
    decision = ("Сбор данных и воспроизводимый прогноз продемонстрированы. "
                + ("На этом примере порядок выхода предсказан верно, но разрешение не проверено. "
                   if pair['elution_order_correct'] else
                   "На этом примере модель ошиблась в порядке выхода веществ. ")
                + "Использовать этот результат как доказательство готовности подбирать "
                "метод разделения пары пока нельзя.")
    coverage = (f"Проверка на всех найденных целевых записях, без отбора удачных: "
                f"{result['benchmark']['target_observations']} наблюдений; "
                f"средняя абсолютная ошибка {result['benchmark']['mae_min']:.2f} мин. "
                "Эти наблюдения связаны между собой системами; это небольшой целевой тест.")
    achieved = ("Прототип определяет точную форму вещества, находит реальные измерения и условия "
                "с источниками, строит прогноз времени удерживания для двух молекул и позволяет "
                "сравнить его с независимыми целевыми значениями. Численные результаты и ограничения "
                "сохраняются воспроизводимо.")
    gaps = ("Пока не подтверждены: разделение смеси, Rs, форма/ширина пиков, точность на конкретной "
            "колонке заказчика и оптимальность условий. Следующий этап — получить из открытых "
            "первичных данных ширины пиков/хроматограммы, завершить проверку метаданных и только "
            "после этого ранжировать методики по качеству разделения. Наблюдаемые времена "
            "в открытой системе — отправная точка, не готовая валидированная методика.")
    support = (f"В обучающей подвыборке есть {a['training_rows_same_system']} других измерений "
               f"этой системы. Максимальное сходство Morgan/Tanimoto (2048 бит) с тенофовиром "
               f"составляет {a['max_local_tanimoto_2048bit']:.2f}, с эмтрицитабином — "
               f"{b['max_local_tanimoto_2048bit']:.2f}. Это диагностический признак, не вероятность точности. "
               "Текущий CatBoost использует шесть молекулярных дескрипторов, fingerprint и условия; "
               "pH-зависимый профиль заряда ещё не включён в признаки этой ML-модели.")
    anomaly = ("В исходной таблице при 13 и 15 мин доли A+B составляют 99,91%, а не 100%. "
               "Значения сохранены без исправления и требуют сверки с первоисточником. "
               "Программа приведена для аудита, не как готовое задание прибору.")
    gradient = [["Время, мин", "A, %", "B, %", "Поток, мл/мин"]] + [
        [p["t [min]"], p["A [%]"], p["B [%]"], p["flow rate [ml/min]"]] for p in pair["gradient_program"]]
    source = pair["source_info"][0]["url"]
    paragraphs = [("Задача", intro), ("Вывод проверки", decision), ("Проверка модели", protocol), ("Условия", method),
                  ("Результат", assessment), ("Проверка без отбора удачных примеров", coverage),
                  ("Поддержка обучающими данными", support), ("Контроль исходных данных", anomaly),
                  ("Что уже работает", achieved), ("Что ещё нужно", gaps)]
    md = f"# {title}\n\n" + "\n\n".join(f"## {h}\n\n{p}" for h, p in paragraphs)
    for heading, table in [("Измерения и прогноз", comparison), ("Программа метода", gradient)]:
        md += f"\n\n## {heading}\n\n| " + " | ".join(table[0]) + " |\n|" + "---|" * len(table[0]) + "\n"
        md += "\n".join("| " + " | ".join(row) + " |" for row in table[1:])
    md += f"\n\nИсточник метода: {source}\n\nЧисленные исходные записи: {a['source_url']}\n"
    (out / "REPORT.md").write_text(md)
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    doc.add_heading(title, 0)
    for heading, paragraph in paragraphs:
        doc.add_heading(heading, level=1)
        doc.add_paragraph(paragraph)
        if heading == "Результат":
            table = doc.add_table(rows=0, cols=4)
            table.style = "Light Shading Accent 1"
            for line in comparison:
                for cell, value in zip(table.add_row().cells, line):
                    cell.text = value
    doc.add_heading("Полная программа метода", level=1)
    table = doc.add_table(rows=0, cols=4)
    table.style = "Light Shading Accent 1"
    for line in gradient:
        for cell, value in zip(table.add_row().cells, line):
            cell.text = value
    doc.add_paragraph("Источник метода: " + source)
    doc.add_paragraph("Численные данные: " + a["source_url"])
    doc.save(out / "Отчёт_тенофовир_эмтрицитабин.docx")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/public_retention/report/observations.jsonl"))
    p.add_argument("--out", type=Path, default=Path("reports/tenofovir_demo"))
    p.add_argument("--train-rows", type=int, default=12000)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--refresh-pubchem", action="store_true")
    build(p.parse_args())
