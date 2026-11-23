"""Inventory local public data for LSS calibration without treating missing gradients as isocratic."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def program_kind(program):
    if len(program) < 2:
        return "incomplete_program"
    try:
        times = [float(p["t [min]"]) for p in program]
        mixes = [tuple(float(p[f"{c} [%]"]) for c in "ABCD") for p in program]
        flows = [float(p["flow rate [ml/min]"]) for p in program]
        if (any(not math.isfinite(v) for v in times+flows) or times[0] != 0
                or any(b <= a for a,b in zip(times,times[1:]))
                or any(f <= 0 for f in flows)
                or any(any(not math.isfinite(v) or not 0 <= v <= 100 for v in mix)
                       or abs(sum(mix)-100) > .01 for mix in mixes)):
            return "invalid_program"
        return "constant_composition_candidate" if len(set(mixes)) == 1 else "gradient"
    except (KeyError, TypeError, ValueError):
        return "invalid_program"


def audit(root):
    datasets = {}
    for path in sorted(root.glob("*/observations.jsonl")):
        system_file = path.with_name("systems.json")
        systems = json.loads(system_file.read_text()) if system_file.exists() else {}
        counts, system_ids, candidates = Counter(), set(), []
        for line in path.open():
            row = json.loads(line)
            system_ids.add(row["system_id"])
            shared = systems.get(row["system_id"], {})
            meta = {**shared.get("metadata", {}), **row.get("metadata", {})}
            program = row.get("gradient_program", shared.get("gradient_program", []))
            kind = program_kind(program)
            counts["observations"] += 1
            counts[kind] += 1
            try:
                t0 = float(meta.get("column.t0"))
                if math.isfinite(t0) and t0 > 0:
                    counts["positive_t0_field_unverified_origin"] += 1
            except (TypeError, ValueError):
                pass
            if kind == "constant_composition_candidate":
                candidates.append({"observation_id": row["observation_id"],
                                   "system_id": row["system_id"], "source_url": row.get("source_url"),
                                   "status": "needs_review_not_calibration_approved"})
        datasets[path.parent.name] = {"counts": dict(counts), "systems": len(system_ids),
            "observation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "systems_sha256": hashlib.sha256(system_file.read_bytes()).hexdigest() if system_file.exists() else None,
            "constant_composition_candidates": candidates}
    return {"datasets": datasets, "scope": "Locally normalized public datasets, not all available literature",
            "status": "NO_AUTOMATICALLY_APPROVED_CALIBRATION_SERIES",
            "limitations": ["A t0 field is not evidence of a measured void time.",
                "Constant pump composition is only a candidate; source review, exact identity, fixed pH/recipe/column and measured t0 remain required.",
                "No widths, Rs or gradient retention predictions are fabricated."],
            "selected_gradient": "Preserved; this audit does not modify or judge the selected gradient."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/processed/public_retention"))
    p.add_argument("--out", type=Path, default=Path("reports/physical_retention"))
    args = p.parse_args()
    result = audit(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["# Пригодность открытых данных для физической модели удерживания", "",
             "Выбранный градиент сохранён. Аудит не оценивает его правильность.", "",
             "| Набор | Наблюдения | Градиент | Нет полной программы | Не прошли проверку состава/времени/потока | Постоянный состав: кандидаты |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, data in result["datasets"].items():
        c = data["counts"]
        lines.append(f"| {name} | {c.get('observations',0)} | {c.get('gradient',0)} | {c.get('incomplete_program',0)} | {c.get('invalid_program',0)} | {c.get('constant_composition_candidate',0)} |")
    lines += ["", "Число в поле t₀ не подтверждает измерение. Одноточечная программа не подтверждает изократику.",
              "Не прошедшие проверку программы могут содержать пропуски потока: это не означает, что опубликованный метод физически неверен. Отсутствующие значения автоматически не восстанавливались.",
              "Ни одна серия автоматически не допущена к калибровке. Это ограничение локального нормализованного корпуса, а не вывод об отсутствии данных в литературе.",
              "", "Для калибровки нужны: точная молекула, одна фиксированная система, не менее трёх разных составов, измеренные t₀/tR с источником. Дополнительные составы следует заранее оставить для независимой проверки.",
              "Для выбранного градиента нужна отдельная градиентная модель и данные о задержке градиента. Изократическую формулу напрямую к нему не применяем."]
    (args.out / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v["counts"] for k,v in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
