#!/usr/bin/env python3
"""Fit the nearest-analog predictor and report leave-one-out accuracy.

Leave-one-out (LOO) is the only honest evaluation at this sample size: for each
molecule, predict its conditions using every *other* molecule, then compare.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from model import NearestAnalogPredictor


def leave_one_out(model: NearestAnalogPredictor, rows: list[dict]) -> dict:
    numeric_errors: dict[str, list[float]] = {t: [] for t in model.numeric_targets}
    categorical_hits: dict[str, list[int]] = {t: [] for t in model.categorical_targets}

    for row in rows:
        result = model.predict_from_fingerprint(
            row["fingerprint"], exclude_source=row["source_file"]
        )
        predicted = result["prediction"]
        actual = row["targets"]

        for target in model.numeric_targets:
            if actual.get(target) is not None and predicted.get(target) is not None:
                numeric_errors[target].append(abs(actual[target] - predicted[target]))

        for target in model.categorical_targets:
            if actual.get(target) is not None and predicted.get(target) is not None:
                categorical_hits[target].append(int(actual[target] == predicted[target]))

    numeric_report = {
        target: {
            "MAE": round(statistics.mean(errors), 2) if errors else None,
            "n": len(errors),
        }
        for target, errors in numeric_errors.items()
    }
    categorical_report = {
        target: {
            "accuracy": round(statistics.mean(hits), 2) if hits else None,
            "n": len(hits),
        }
        for target, hits in categorical_hits.items()
    }
    return {"numeric": numeric_report, "categorical": categorical_report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/dataset.json")
    parser.add_argument("--model-out", default="models/nearest_analog.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    rows = data["rows"]
    if len(rows) < 3:
        raise SystemExit(f"Need at least 3 labelled molecules, have {len(rows)}")

    model = NearestAnalogPredictor(
        numeric_targets=data["numeric_targets"],
        categorical_targets=data["categorical_targets"],
        k=args.k,
    ).fit(rows)

    report = leave_one_out(model, rows)
    print(f"Training molecules: {len(rows)}  |  k={args.k}\n")
    print("Numeric targets (mean absolute error):")
    for target, stats in report["numeric"].items():
        mae = stats["MAE"]
        print(f"  {target:24s} MAE={mae if mae is not None else '  -'}  (n={stats['n']})")
    print("\nCategorical targets (accuracy):")
    for target, stats in report["categorical"].items():
        acc = stats["accuracy"]
        print(f"  {target:24s} acc={acc if acc is not None else ' -'}  (n={stats['n']})")

    model.save(args.model_out)
    print(f"\nModel saved -> {args.model_out}")


if __name__ == "__main__":
    main()
