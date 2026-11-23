#!/usr/bin/env python3
"""Compare public RP methods for TFV/FTC; report uncertainty, not invented Rs."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from open_retention import load_observations, number
from realistic_features import ChargeCache, enriched_vector
from train_open_retention import vector
from propose_pair_method import SPEC, check_method, possible_gap
from curate_target_evidence import identity_status


def method_issues(system):
    """Screen data integrity, NOT instrument/manufacturer compatibility."""
    m, program = system["metadata"], system["gradient_program"]
    issues = []
    if not m.get("column.name"):
        issues.append("missing_column")
    label = " ".join(str(info.get("name", "")) for info in system.get("source_info", [])).lower().replace("_", " ")
    column_name = str(m.get("column.name", "")).lower()
    for family in ("raptor biphenyl", "cortecs", "hss t3", "hss c18", "xb-c18", "beh c18", "ascentis", "synergi"):
        if family in label and family not in column_name:
            issues.append("source_method_label_disagrees_with_column_metadata")
    for key in ["column.length", "column.id", "column.particle.size", "column.temperature", "eluent.A.pH"]:
        value = number(m.get(key))
        if value is None or value <= 0:
            issues.append("missing_or_invalid_" + key)
    if not program:
        return issues + ["missing_gradient"]
    previous = -1.0
    solvents = ["h2o", "meoh", "acn", "iproh", "acetone", "hex", "chcl3", "ch2cl2", "hept"]
    for p in program:
        t, flow = number(p.get("t [min]")), number(p.get("flow rate [ml/min]"))
        if t is None or t <= previous:
            issues.append("invalid_gradient_time")
        if t is not None:
            previous = t
        if flow is None or flow <= 0:
            issues.append("missing_or_invalid_flow")
        fractions = [number(p.get(f"{c} [%]")) for c in "ABCD"]
        if any(v is None or not 0 <= v <= 100 for v in fractions):
            issues.append("invalid_channel_fractions")
            continue
        if abs(sum(fractions)-100) > 1e-6:
            issues.append("gradient_mass_balance")
        for c, fraction in zip("ABCD", fractions):
            if fraction == 0:
                continue
            values = [number(m.get(f"eluent.{c}.{s}")) for s in solvents]
            if any(v is not None and v < 0 for v in values) or abs(sum(v or 0 for v in values)-100) > 1e-6:
                issues.append("active_channel_solvent_balance")
    # Only this exact product/geometry has manufacturer constraints checked locally.
    if not issues and m.get("column.name") == SPEC["name"]:
        issues.extend(check_method(m, program))
    return sorted(set(issues))


def select_model(paths):
    evaluations = [(path, json.loads(path.read_text())) for path in paths]
    reference = evaluations[0][1]
    for _, r in evaluations[1:]:
        if r["dataset_sha256"] != reference["dataset_sha256"]:
            raise ValueError("Models use different dataset revisions")
        for split in ("selection", "calibration", "test"):
            if r["split_observation_ids"][split] != reference["split_observation_ids"][split]:
                raise ValueError("Models do not share frozen holdouts")
    choices = [(path, r, name) for path, r in evaluations for name in r["models"]]
    return min(choices, key=lambda item: item[1]["models"][item[2]]["selection"]["macro_group_mae_min"]), evaluations


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    (path, evaluation, model_name), evaluations = select_model(args.evaluations)
    data_digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    systems_path = args.data.parent / "systems.json"
    systems_digest = hashlib.sha256(systems_path.read_bytes()).hexdigest()
    if data_digest != evaluation["dataset_sha256"] or systems_digest != evaluation["systems_sha256"]:
        raise ValueError("Data/method metadata differ from model manifest")
    systems = json.loads(systems_path.read_text())
    observations = load_observations(args.data)
    by_id = {r["observation_id"]: r for r in observations}
    training = [by_id[i] for i in evaluation["split_observation_ids"]["train"]]
    counts = Counter(r["system_id"] for r in training)
    demo = json.loads(Path("reports/tenofovir_demo/result.json").read_text())
    smiles = {name: info["payload"]["PropertyTable"]["Properties"][0]["SMILES"] for name, info in demo["pubchem"].items()}
    model_path = path.parent / (model_name + ".cbm")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    cache = ChargeCache(args.cache)
    def features(row):
        return (vector(row, evaluation["numeric_metadata_features"]) if model_name == "baseline" else
                enriched_vector(row, evaluation["numeric_metadata_features"], cache, evaluation["feature_version"]))
    pools, rejected = defaultdict(list), []
    for sid, system in systems.items():
        if not system["source_info"] or system["source_info"][0].get("method.type") != "RP":
            continue
        issues = method_issues(system)
        if issues:
            rejected.append({"system_id": sid, "column": system["metadata"].get("column.name"), "issues": issues})
        elif counts[sid] >= args.minimum_training_rows or (system["metadata"].get("column.name") == SPEC["name"] and counts[sid] > 0):
            pools[system["metadata"]["column.name"]].append((sid, system))
    columns = sorted(pools, key=lambda col: (-sum(counts[sid] for sid, _ in pools[col]), col))[:args.columns]
    if SPEC["name"] in pools and SPEC["name"] not in columns:
        columns[-1] = SPEC["name"]  # Include the previously studied, passport-checked reference.
    if len(columns) < 2:
        raise ValueError("Too few columns with complete public methods")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {r["observation_id"]: generator.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in training}
    target_fp = {name: generator.GetFingerprint(Chem.MolFromSmiles(s)) for name, s in smiles.items()}
    candidates = []
    for col in columns:
        # Deterministic, coverage-based shortlist, not chosen by target experimental outcome.
        methods = sorted(pools[col], key=lambda item: (-counts[item[0]], item[0]))[:3]
        local = {r["compound_id"]: r for r in training if r["metadata"].get("column.name") == col}
        neighborhood = {}
        for name, target in target_fp.items():
            similarities = DataStructs.BulkTanimotoSimilarity(target, [fp[r["observation_id"]] for r in local.values()])
            neighborhood[name] = {"unique_full_inchikeys_same_column_product": len(local),
                                  "fp_gt_0_3": sum(v > .3 for v in similarities),
                                  "fp_gt_0_2": sum(v > .2 for v in similarities),
                                  "max_tanimoto": max(similarities) if similarities else None,
                                  "scope": "same product name, not necessarily same geometry/conditions"}
        for sid, original in methods:
            for scale in (.8, 1.0, 1.2):
                system = deepcopy(original)
                for p in system["gradient_program"]:
                    p["t [min]"] = str(round(float(p["t [min]"])*scale, 5))
                issues = method_issues(system)
                if issues:
                    continue
                estimate = {name: max(0, float(np.expm1(model.predict([features({**system, "smiles": structure})])[0])))
                            for name, structure in smiles.items()}
                radius = evaluation["models"][model_name]["intervals_90_nominal"]["radius_min"]
                bounds = {name: [max(0, value-radius), value+radius] for name, value in estimate.items()}
                m = system["metadata"]
                candidates.append({"candidate_id": f"{sid}:time_x{scale}", "source_system": sid, "column": col,
                                   "conditions": m, "gradient_program": system["gradient_program"],
                                   "origin": "source_program" if scale == 1 else "generated_time_scaled_program",
                                   "time_scale": scale, "predicted_tr_min": estimate, "individual_interval_90_nominal_min": bounds,
                                   "time_gap_min": abs(estimate["tenofovir"]-estimate["emtricitabine"]),
                                   "possible_gap_min": possible_gap(bounds["tenofovir"], bounds["emtricitabine"]),
                                   "cycle_time_min": float(system["gradient_program"][-1]["t [min]"]),
                                   "same_system_training_rows": counts[sid], "local_coverage": neighborhood,
                                   "method_source": system["source_info"],
                                   "metadata_source": demo["target_predictions"][0]["source_url"].split("/processed_data/")[0] +
                                                      f"/processed_data/{sid.split(':')[1]}/{sid.split(':')[1]}_metadata.tsv",
                                   "predicted_rs": None, "pressure_bar": None, "execution_ready": False,
                                   "manufacturer_limits_checked": col == SPEC["name"],
                                   "warnings": ["No peak width/Rs model", "Dwell volume and pressure unknown",
                                                "Point predictions do not establish separation",
                                                "90% nominal intervals are not validated on generated programs"]})
    if not candidates:
        raise ValueError("No candidates")
    # A numerical surrogate is exported but is never promoted to a verified separation rank.
    candidates.sort(key=lambda r: (-r["time_gap_min"], r["cycle_time_min"], r["candidate_id"]))
    for index, r in enumerate(candidates, 1):
        r["surrogate_rank_not_separation_rank"] = index
    benchmark = []
    for record in evaluation["target_diagnostic"][model_name]["records"]:
        row = by_id[record["observation_id"]]
        target_name = next(name for name, info in demo["pubchem"].items()
                           if row["compound_id"].split("-")[0] == info["payload"]["PropertyTable"]["Properties"][0]["InChIKey"].split("-")[0])
        benchmark.append({**record, "column": row["metadata"].get("column.name"),
                          "target_name": target_name, "identity_status": identity_status(row["compound_id"], target_name),
                          "source_url": row["source_url"], "absolute_error_min": abs(record["measured_min"]-record["predicted_min"])})
    pair_benchmarks = []
    for pair in demo["pairs"]:
        groups = {name: [r for r in benchmark if r["system_id"] == pair["system_id"] and
                        by_id[r["observation_id"]]["compound_id"].startswith(key)]
                  for name, key in {"tenofovir": "SGOIRFVFHAKUTI", "emtricitabine": "XQSPYNMVSIKCOC"}.items()}
        if all(len(v) == 1 and v[0]["identity_status"] == "exact_full_inchikey" for v in groups.values()):
            a, b = groups["tenofovir"][0], groups["emtricitabine"][0]
            pair_benchmarks.append({"system_id": pair["system_id"], "column": a["column"],
                                    "measured_tfv_min": a["measured_min"], "measured_ftc_min": b["measured_min"],
                                    "predicted_tfv_min": a["predicted_min"], "predicted_ftc_min": b["predicted_min"],
                                    "order_correct": (a["measured_min"] < b["measured_min"]) == (a["predicted_min"] < b["predicted_min"]),
                                    "independent": pair["system_id"] != "report:0209",
                                    "note": "0209 duplicates METLIN 0186; same method does not prove same injection",
                                    "rs": None})
    result = {"target": "tenofovir TFV + emtricitabine FTC; not TDF/TAF", "selected_evaluation": str(path),
              "selected_model": model_name, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
              "selection_basis": "lowest molecule-macro MAE on identical frozen selection set; not target results",
              "evaluations": [{"path": str(p), "data": r} for p, r in evaluations],
              "candidate_count": len(candidates), "column_count": len(columns), "columns": columns,
              "candidates": candidates, "rejected_methods": rejected, "target_benchmark": benchmark,
              "pair_benchmark": pair_benchmarks, "selected_model_metrics": evaluation["models"][model_name],
              "selected_new_source_metrics": evaluation["new_source_test"],
              "rank_status": "UNRESOLVED_WITHOUT_PEAK_WIDTHS_AND_VALIDATED_UNCERTAINTY",
              "column_comparison_scope": "column+method packages, NOT isolated causal column effects",
              "execution_ready": False, "separation_confirmed": False}
    exact = [r for r in benchmark if r["identity_status"] == "exact_full_inchikey" and r["system_id"] != "report:0209"]
    result["identity_checked_diagnostic"] = {
        "rows": len(exact), "mae_min": float(np.mean([r["absolute_error_min"] for r in exact])),
        "observation_ids": [r["observation_id"] for r in exact],
        "scope": "Full InChIKey match; METLIN alias 0209 removed; other primary-label review still required"}
    result["model_disagreement_diagnostic"] = {
        "historical_target_mae_min": {str(p): r["target_diagnostic"][r["selected_model"]]["mae_min"] for p, r in evaluations},
        "promotion_to_production": False,
        "reason": "Global holdout improvement is insufficient; target accuracy and separation are not validated"}
    from recommend_lab import choose_experiments, write_packet
    result["lab_recommendation"] = choose_experiments(result, method_issues)
    write_packet(args.out, result["lab_recommendation"])
    cache.save()
    (args.out / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    flat = [{"candidate_id": r["candidate_id"], "column": r["column"],
             "tfv_predicted_min": r["predicted_tr_min"]["tenofovir"], "ftc_predicted_min": r["predicted_tr_min"]["emtricitabine"],
             "gap_surrogate_min": r["time_gap_min"], "cycle_min": r["cycle_time_min"], "origin": r["origin"],
             "pH_A_source": r["conditions"]["eluent.A.pH"], "same_system_training_rows": r["same_system_training_rows"],
             "ranking_status": result["rank_status"]} for r in candidates]
    with (args.out / "candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    active_csvs = {r["source_system"].replace(":", "_") + "_gradient.csv" for r in candidates if r["time_scale"] == 1}
    # Preserve superseded generated exports, but keep them out of the current shortlist.
    for old in args.out.iterdir():
        if not re.fullmatch(r"report_\d+_gradient\.csv", old.name) or old.name in active_csvs:
            continue
        archive = args.out / "superseded_gradients"
        archive.mkdir(exist_ok=True)
        target = archive / old.name
        version = 1
        while target.exists():
            target = archive / (old.stem + f"_v{version}.csv")
            version += 1
        old.rename(target)
    for r in candidates:
        if r["time_scale"] != 1:
            continue
        with (args.out / (r["source_system"].replace(":", "_") + "_gradient.csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(r["gradient_program"][0])); writer.writeheader(); writer.writerows(r["gradient_program"])
    write_report(args.out, result)
    print(json.dumps({"columns": columns, "candidates": len(candidates), "model": str(model_path),
                      "first_lab_experiment": result["lab_recommendation"]["primary"]["candidate"]["candidate_id"]
                      if result["lab_recommendation"]["primary"] else None,
                      "target_mae_min": evaluation["target_diagnostic"][model_name]["mae_min"],
                      "report": str(args.out / "Отчёт_для_заказчика.docx")}, ensure_ascii=False, indent=2))


def write_report(out, result):
    from docx import Document
    from docx.shared import Inches, Pt
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = result["selected_model_metrics"]
    source = result["selected_new_source_metrics"]
    radius = m["intervals_90_nominal"]["radius_min"]
    bench = result["target_benchmark"]
    target_mae = float(np.mean([r["absolute_error_min"] for r in bench]))
    identity_checked = [r for r in bench if r["identity_status"] == "exact_full_inchikey" and r["system_id"] != "report:0209"]
    checked_mae = float(np.mean([r["absolute_error_min"] for r in identity_checked]))
    independent_pairs = [r for r in result["pair_benchmark"] if r["independent"]]
    correct = sum(r["order_correct"] for r in independent_pairs)
    sections = []
    def add(title, text):
        sections.append((title, text))
    lab = result.get("lab_recommendation", {})
    if lab.get("primary"):
        first = lab["primary"]["candidate"]; meta = first["conditions"]
        add("Рекомендуемый первый эксперимент", f"{first['column']}, {meta['column.length']} × {meta['column.id']} мм, "
            f"{meta['column.particle.size']} мкм; исходная программа {first['source_system']}, pH A по источнику {meta['eluent.A.pH']}, "
            f"{meta['column.temperature']} °C. Выбран по полноте источника и известным паспортным ограничениям, "
            "не как доказанный ML-оптимум. Полные составы фаз, градиент, предложенные параметры ввода/детекции, "
            "два запасных варианта и обязательные проверки приведены в «Задание_в_лабораторию.docx» и LAB_PLAN.md. "
            "Передать аналитику для подготовки скрининга; запуск требует его проверки и согласования. Разделение не подтверждено.")
    else:
        add("Рекомендуемый первый эксперимент", "Блокер: нет полного прослеживаемого исходного метода. "
            "Причины сохранены в lab_recommendation.json; автоматически достроенный рецепт не предлагается.")
    add("Резюме", f"Построен и проверен исследовательский прототип на открытых данных. Выполнено компьютерное сравнение "
        f"{result['candidate_count']} программ на {result['column_count']} продуктах колонок. Это не лабораторные испытания. "
        "Система прогнозирует времена удерживания, сохраняет происхождение условий и исключает неполные/противоречивые методы. "
        "Доказанного оптимума и подтверждённого разделения TFV/FTC пока нет: модель ширин пиков и Rs отсутствует.")
    add("Постановка задачи", "Вход: химические структуры TFV и FTC, колонка и программа ВЭЖХ. Выход текущей модели: tR каждого "
        "вещества в минутах и диапазон неопределённости. Целевая промышленная задача шире: подобрать воспроизводимое разделение "
        "смеси при допустимых давлении, времени и расходе растворителя. TFV нельзя подменять пролекарствами TDF/TAF. "
        "Пользователь подтвердил целевую пару TFV + эмтрицитабин; именно она рассматривается в отчёте.")
    add("Использованные данные", "RepoRT: 183 818 наблюдений в 421 системе; RP-поднабор — 173 949 наблюдений. "
        "METLIN SMRT и U Liège собраны отдельно; их нельзя просто суммировать с RepoRT из-за пересечений. "
        "Для целевых веществ найдены 21 наблюдение и 7 общих систем. Общий метод не подтверждает совместную инъекцию. "
        "Собраны 306 реальных QC-сигналов с предварительными ширинами/Rs, но это другая смесь, не доказательство разделения TFV/FTC. "
        "Извлечены 67 таблиц из 15 разрешённых к повторному использованию статей. Данные заказчика не использованы.")
    add("Контроль качества источников", "Таблицы вне основного текста JATS теперь извлекаются. В PMC12114340 времена TFV/FTC "
        "в тексте (5,08/6,21 мин) противоречат таблице 5 (0,80/0,78 мин), включая порядок выхода. Обе версии изолированы. "
        "13 опытов PMC3929521 относятся к FTC/ламивудину, а не TFV/FTC; условия pH и объёма ввода противоречивы. "
        "Повтор METLIN в RepoRT 0209/0186 не считается независимым подтверждением. "
        "Программы с нарушением баланса фаз не исправлялись молча, а исключались из генерации кандидатов. "
        "Также обнаружено несовпадение названий CORTECS/Raptor Biphenyl между меткой метода и полем колонки: "
        "такие методы исключены из набора рекомендаций до сверки. Полный обучающий корпус ещё требует такого же "
        "первичного аудита; результаты его моделей не являются результатами обучения на полностью выверенной базе.")
    add("Дополнительная проверка идентичности молекул", "Из 21 кандидата RepoRT только 17 совпадают с целевыми веществами по "
        "полному InChIKey. Две записи FTC имеют другой стереохимический блок, две записи TFV не задают стереохимию. "
        "Они не считаются подтверждёнными измерениями нужных форм. Поэтому общая историческая диагностика на 21 строке "
        "показана отдельно от строгой проверки идентичности. Это различие важно: одинаковая связность не гарантирует ту же молекулу.")
    add("Как устроено прогнозирование", "RDKit: дескрипторы, функциональные группы и fingerprints; Dimorphite-DL: возможные "
        "зарядовые формы при pH метода (не измеренный pKa и не доли форм); CatBoost: регрессия log(1+tR). "
        "Учитываются геометрия, составы предварительно смешанных фаз, градиент, температура и гидродинамические признаки. "
        "RAG/поиск источников обеспечивает реальные методы и проверяемые ссылки. Языковая модель объясняет результаты, "
        "но не заменяет численный прогноз и эксперимент. log(k) не рассчитывается без достоверного t0.")
    add("Обучение и проверка", "Добавлены 18 000 обучающих наблюдений с последовательным обходом систем для улучшения их "
        "покрытия. Исходные выборки выбора модели (1 119), калибровки (1 800) и проверки (1 881 строка) заморожены. "
        "Молекулы контрольных выборок исключены из расширения обучения; TFV/FTC/TDF/TAF не используются при обучении. "
        "Выбор среди вариантов выполнен по macro-MAE на выборке выбора, а не по результату TFV/FTC. "
        "Контрольные примеры уже анализировались при разработке: это исследовательская проверка, не нетронутая финальная валидация.")
    add("Почему больше данных не означает лучше для целевой смеси", "В расширенном эксперименте общая MAE уменьшилась "
        "с 1,485 до 1,407 мин, но MAE TFV/FTC увеличилась с 1,485 до 2,294 мин. Ошибка на новых источниках также выросла "
        "с 2,81 до 3,29 мин. Это отрицательный результат для переноса на целевую задачу, который нельзя скрывать за "
        "улучшением общей метрики. Дополнительно проверено обучение с весами по структурной близости к TFV/FTC "
        "без использования их экспериментальных tR. Все варианты, включая неудачные, показаны в приложении.")
    add("Измеренное качество выбранного варианта", f"MAE на фиксированной проверочной выборке: {m['test']['mae_min']:.3f} мин; "
        f"macro-MAE: {m['test']['macro_group_mae_min']:.3f} мин. Историческая диагностика по связности (21 строка): {target_mae:.3f} мин. "
        f"После полного InChIKey и исключения копии METLIN: {len(identity_checked)} строк, MAE {checked_mae:.3f} мин. "
        f"Правильный порядок выхода: {correct} из {len(independent_pairs)} однозначных пар после исключения повтора METLIN. "
        "Это слишком мало для оценки вероятности успешного разделения. "
        f"Радиус номинального 90%-интервала: {radius:.2f} мин; фактическое покрытие молекулярных групп: "
        f"{100*m['intervals_90_nominal']['all_rows_in_group_coverage']:.1f}%. "
        f"Отдельный тест новых источников: MAE {source['mae_min']:.2f} мин, покрытие {100*source['intervals_90_nominal']['all_rows_in_group_coverage']:.1f}%. "
        "Эти интервалы не гарантируют такое же покрытие на произвольной новой колонке или сгенерированном градиенте.")
    add("Как сравнивались колонки", "Выбраны продукты с полными условиями и покрытием в обучении. Для каждого использовано до "
        "трёх открытых методов и масштабирование времени 0,8/1,0/1,2. Составы фаз не подменялись универсальным рецептом. "
        "Сравниваются пакеты «колонка + программа», а не изолированное влияние сорбента. Геометрия и фазы находятся в приложении "
        "и comparison.json; исходные градиенты — в CSV. Проверка допустимых режимов производителя завершена только для "
        "указанной геометрии Acclaim. Остальные кандидаты требуют проверки паспортов; ни один не разрешён к автоматическому запуску.")
    add("Что можно рекомендовать сейчас", "Использовать результаты как обоснованный список условий для последующего скрининга. "
        "Численное ранжирование по разнице прогнозных tR сохранено только как суррогат: оно не является ранжированием по Rs. "
        "При пересекающихся диапазонах прогнозов нельзя объявлять лучший метод или порядок выхода. "
        "HILIC, ионообменные и другие режимы этой RP-моделью не оценивались.")
    add("План экспериментальной проверки", "До запуска специалист подтверждает химическую форму, матрицу, точный артикул "
        "колонки, допустимые pH/растворители/давление и dwell volume прибора. Затем: холостой раствор, отдельные стандарты, "
        "смесь и повторные инъекции в выбранных совместимых условиях. Фиксируются tR, ширины пиков, измеренный Rs, "
        "форма пика, давление, температура и воспроизводимость. Условия/пороги пригодности согласуются с аналитиком под задачу; "
        "tR сам по себе не является критерием успешного разделения. После первых опытов можно применять байесовскую "
        "оптимизацию с ограничениями по измеренному качеству, времени и расходу. Она не заменяет измерения.")
    add("Вывод для заказчика", "Получена воспроизводимая система предварительного отбора ВЭЖХ-условий из открытых данных "
        "с прогнозом удерживания, контролем ошибок источников и явными границами применимости. "
        "Для утверждения о полноценной автоматической разработке метода необходимо подтвердить разделение смеси и перенос "
        "на конкретный прибор. Текущий результат — исследовательский shortlist, а не валидированная методика.")
    md = "# TFV + FTC: сравнение колонок и открытых методов\n\nДата: 23 октября 2026. Исследовательский отчёт.\n\n"
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Тенофовир + эмтрицитабин", 0)
    doc.add_paragraph("Сравнение колонок и открытых методов · 23 октября 2026\nИсследовательский отчёт, не валидированная методика")
    for title, text in sections:
        md += f"## {title}\n\n{text}\n\n"
        doc.add_heading(title, level=1); doc.add_paragraph(text)
    def table(title, headers, rows):
        nonlocal md
        doc.add_heading(title, level=1)
        t = doc.add_table(rows=1, cols=len(headers)); t.style = "Light Shading Accent 1"
        for cell, value in zip(t.rows[0].cells, headers): cell.text = str(value)
        for row in rows:
            for cell, value in zip(t.add_row().cells, row): cell.text = str(value)
        md += f"## {title}\n\n| " + " | ".join(headers) + " |\n|" + "---|"*len(headers) + "\n"
        md += "\n".join("| " + " | ".join(map(str,row)) + " |" for row in rows) + "\n\n"
    rows = []
    for entry in result["evaluations"]:
        r = entry["data"]
        for name, metrics in r["models"].items():
            rows.append([Path(entry["path"]).parent.name + "/" + name, r["split_sizes"]["train"],
                         f"{metrics['selection']['macro_group_mae_min']:.3f}", f"{metrics['test']['mae_min']:.3f}",
                         f"{r['target_diagnostic'][name]['mae_min']:.3f}"])
    table("Приложение 1. Сравнение моделей", ["Эксперимент/признаки", "Обучение, строк", "Выбор: macro-MAE", "Тест: MAE", "TFV/FTC: MAE"], rows)
    doc.add_paragraph("Столбец TFV/FTC в этой таблице — историческая диагностика 21 записи по связности, "
                      "не независимая валидация точных стереохимических форм. Строгий пересчёт выбранной модели описан выше.")
    md += "Столбец TFV/FTC — историческая диагностика 21 записи по связности; строгий пересчёт описан выше.\n\n"
    representatives = []
    for col in result["columns"]:
        # Original source program with greatest coverage, not the largest predicted gap.
        representatives.append(sorted([r for r in result["candidates"] if r["column"] == col and r["time_scale"] == 1],
                                      key=lambda r: (-r["same_system_training_rows"], r["candidate_id"]))[0])
    rows = []
    for r in representatives:
        p = r["predicted_tr_min"]
        m = r["conditions"]
        rows.append([r["column"] + f" ({r['source_system']})", f"{m['column.length']}×{m['column.id']}; {m['column.particle.size']}",
                     m["eluent.A.pH"], f"{p['tenofovir']:.2f}", f"{p['emtricitabine']:.2f}", f"{r['time_gap_min']:.2f}"])
    table("Приложение 2. Прогнозы исходных программ — не измерения", ["Колонка / система", "мм; мкм", "pH A", "TFV, мин", "FTC, мин", "ΔtR, не Rs"], rows)
    doc.add_paragraph(f"Для каждой оценки указан общий калиброванный радиус {radius:.2f} мин (граница снизу — ноль). "
                      "Не трактовать две цифры после запятой как достигнутую точность. Полные интервалы — в comparison.json.")
    md += f"Калиброванный радиус отдельных оценок: {radius:.2f} мин, нижняя граница обрезана на нуле. Для новых программ покрытие не гарантировано.\n\n"
    rows = [[r["system_id"], r["column"], f"{r['measured_tfv_min']:.2f} / {r['measured_ftc_min']:.2f}",
             f"{r['predicted_tfv_min']:.2f} / {r['predicted_ftc_min']:.2f}", "да" if r["order_correct"] else "НЕТ"]
            for r in independent_pairs]
    table("Приложение 3. Сверка с открытыми измерениями", ["Система", "Колонка", "Опыт TFV/FTC", "Прогноз TFV/FTC", "Порядок верен"], rows)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for name, marker in [("tenofovir", "o"), ("emtricitabine", "s")]:
        values = [r for r in bench if r["target_name"] == name]
        values = [r for r in values if r["identity_status"] == "exact_full_inchikey" and r["system_id"] != "report:0209"]
        ax.scatter([r["measured_min"] for r in values], [r["predicted_min"] for r in values], label=name, marker=marker)
    limit = max(max(r["measured_min"],r["predicted_min"]) for r in bench)*1.1
    ax.plot([0,limit],[0,limit], "--", color="gray", label="perfect prediction")
    ax.set(xlabel="Reference tR (min)", ylabel="Predicted tR (min)", title="TFV/FTC: development diagnostic, not final validation")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "target_predictions.png", dpi=160); plt.close(fig)
    doc.add_heading("Диагностика прогнозов", level=1); doc.add_picture(str(out / "target_predictions.png"), width=Inches(6.1))
    md += "![Прогноз против измерений](target_predictions.png)\n\n"
    doc.add_heading("Источники и воспроизводимость", level=1)
    sources = sorted({r["source_url"] for r in bench} | {r["metadata_source"] for r in representatives})
    sources += ["https://pubmed.ncbi.nlm.nih.gov/16005269/ — первичная работа TFV/FTC на Atlantis dC18; "
                "полный рецепт не восстановлен, в ML-сравнение не включена.",
                "https://eprints.ugd.edu.mk/30820/2/book%20of%20abstract.pdf — работа по нескольким колонкам; "
                "во введении указан TDF, поэтому не использована как подтверждение TFV.",
                "Локальные материалы: comparison.json, candidates.csv, *_gradient.csv, reports/target_data_curation/."]
    for item in sources: doc.add_paragraph(item)
    md += "## Источники и воспроизводимость\n\n" + "\n\n".join(sources) + "\n"
    for r in representatives:
        doc.add_heading(r["column"] + " — " + r["source_system"], level=2)
        meta = r["conditions"]
        active = [c for c in "ABCD" if any(float(p[f"{c} [%]"]) > 0 for p in r["gradient_program"])]
        for c in active:
            phase = {k.removeprefix(f"eluent.{c}."): v for k,v in meta.items()
                     if k.startswith(f"eluent.{c}.") and v not in ("", "0", None)}
            doc.add_paragraph(f"Канал {c}: " + json.dumps(phase, ensure_ascii=False))
        doc.add_paragraph(f"Температура: {meta['column.temperature']} °C. Программа: {r['source_system'].replace(':','_')}_gradient.csv. "
                          "pH — значение источника, способ измерения требует проверки.")
    doc.save(out / "Отчёт_для_заказчика.docx")
    (out / "REPORT.md").write_text(md)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/public_retention/report/observations.jsonl"))
    p.add_argument("--evaluations", type=Path, nargs="+", default=[Path("reports/realism_evaluation/evaluation.json"),
                   Path("reports/hydrodynamics_evaluation/evaluation.json"), Path("reports/expanded_evaluation/evaluation.json"),
                   Path("reports/domain_weighted_evaluation/evaluation.json")])
    p.add_argument("--cache", type=Path, default=Path("data/processed/public_retention/research_cache/charges.json"))
    p.add_argument("--columns", type=int, default=10)
    p.add_argument("--minimum-training-rows", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("reports/customer_multicolumn"))
    run(p.parse_args())
