#!/usr/bin/env python3
"""Summarize downloaded material, derived labels and actual training readiness."""
import json
from pathlib import Path

root = Path("data/external")
destination = Path("reports/open_data_expansion")
destination.mkdir(parents=True, exist_ok=True)


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


inventory = read(root / "chromatogram_inventory.json", [])
downloads = read(root / "chromatogram_downloads.json", {"files": {}})
signals = [r for r in downloads["files"].values() if r["status"] == "verified"]
peaks = read(Path("data/processed/public_retention/chromatograms/report.json"), {})
methods = read(root / "open_methods/manifest.json", {})
benchmarks = read(root / "signal_benchmarks/catalog.json", [])
reported_path = Path("data/processed/public_retention/reported_suitability")
summary = {
    "indexed_raw_signals": len(inventory), "indexed_raw_bytes": sum(r["bytes"] for r in inventory),
    "qc_signals_verified": len(signals), "qc_bytes_verified": sum(r["bytes"] for r in signals),
    "qc_download_failures": sum(r["status"] == "failed" for r in downloads["files"].values()),
    "signals_parsed": peaks.get("processed_signals", 0),
    "derived_peak_width_candidates": peaks.get("derived_widths", 0),
    "derived_rs_candidates": peaks.get("derived_resolution_pairs", 0),
    "signals_linked_to_author_runs": peaks.get("signals_linked_to_author_runs_with_rt_check", 0),
    "approved_peak_training_rows": peaks.get("approved_training_rows", 0),
    "licensed_full_texts": sum(r["status"] == "downloaded" for r in methods.get("articles", [])),
    "papers_excluded_license": sum(r["status"].startswith("excluded_license") for r in methods.get("articles", [])),
    "evidence_candidates": methods.get("evidence_candidates", 0),
    "target_specific_evidence": methods.get("target_specific_candidates", 0),
    "article_tables": sum(len(read(p, [])) for p in (root / "open_methods").glob("*_tables.json")),
    "reported_resolution_values": len(read(reported_path / "reported_resolution.json", [])),
    "signal_benchmarks": [{"source": r["source"], "status": r["status"],
                            "files": len(r.get("files", [])),
                            "archive_members": sum(len(f.get("archive_members", [])) for f in r.get("files", []))}
                           for r in benchmarks],
    "status": "Data collection and provisional signal-derived measurements; not a trained Rs predictor"}
(destination / "summary.json").write_text(json.dumps(summary, indent=2))
text = ("# Расширение открытых данных\n\n"
        f"Полный индекс: {summary['indexed_raw_signals']} сигналов U Liège "
        f"({summary['indexed_raw_bytes']/1e9:.2f} ГБ). Загружено и проверено "
        f"{summary['qc_signals_verified']} QC-сигналов ({summary['qc_bytes_verified']/1e9:.2f} ГБ).\n\n"
        f"Разобрано {summary['signals_parsed']} хроматограмм; получено "
        f"{summary['derived_peak_width_candidates']} кандидатов ширины на половине высоты и "
        f"{summary['derived_rs_candidates']} оценок Rs для соседних QC-компонентов одной инъекции.\n\n"
        f"С авторской таблицей по хронологии и проверке tR связаны {summary['signals_linked_to_author_runs']} инъекций. "
        "Это смесь урацила, 3-цианопиридина и ибупрофена. Идентификация пиков предполагаемая, "
        "по известному составу и диапазонам tR; автоматическая интеграция требует проверки. "
        "Ширина измерена относительно локальной линейной базовой линии; Rs рассчитан по "
        "формуле 1,18·ΔtR/(wh1+wh2). Это не авторские опубликованные значения Rs. "
        "Все новые метки сохраняются с training_ready=false. Отдельного прогноза Rs ещё нет.\n\n"
        f"Лицензированные полные тексты: {summary['licensed_full_texts']}; "
        f"отложены из-за лицензии: {summary['papers_excluded_license']}. "
        f"Извлечены {summary['article_tables']} таблиц и {summary['evidence_candidates']} фрагмент с численными "
        f"параметрами, включая {summary['target_specific_evidence']} упоминаний целевых веществ. "
        "Это контекст для курирования, а не автоматически подтверждённые измерения.\n\n"
        "## Дополнительные наборы сигналов\n\n")
for b in summary["signal_benchmarks"]:
    text += f"- {b['source']}: {b['status']}, {b['archive_members']} файлов внутри архивов.\n"
text += ("\nОни полезны для проверки обработки сигналов, но не доказывают перенос удерживания "
         "на TFV/FTC. Их содержимое ещё не нормализовано в обучающие наблюдения.\n\n"
         f"Из статьи PMC12216858 отдельно сохранены {summary['reported_resolution_values']} авторских значения Rs "
         "(4,73 и 7,32) для FTC/DAC/LDV и 3 строки с tR, tailing, N и k. "
         "Это сводные параметры метода, не новая подтверждённая инъекция. "
         "Значения RSD% при проверке робастности не преобразуются в абсолютный Rs. "
         "Коэффициент k взят из таблицы авторов; измеренный t0 по нему не выдумывается.\n\n"
         "## Что пока отсутствует\n\n"
         "Смешанные хроматограммы TFV/FTC, проверенные близкие/перекрывающиеся пики, "
         "ручная проверка интеграции и независимый тест прогноза Rs. QC-пики широко разнесены, "
         "поэтому они не покрывают сложные разделения. Текущая точность CatBoost этим сбором "
         "не изменена; новые данные не подмешаны в обучение автоматически.\n")
(destination / "REPORT.md").write_text(text)
print(json.dumps(summary, indent=2))
