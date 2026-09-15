#!/usr/bin/env python3
"""Rank an HPLC screening grid with the provisional direct-tR model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor

from features import compute_descriptors
from train_tr_model import featurize


def numbers(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--column", default="C18")
    parser.add_argument("--length", type=float, default=150)
    parser.add_argument("--column-id", type=float, default=4.6)
    parser.add_argument("--particle", type=float, default=5.0)
    parser.add_argument("--modifier", choices=("acetonitrile", "methanol"), default="acetonitrile")
    parser.add_argument("--organic", default="30,40,50")
    parser.add_argument("--ph", default="3.0,3.5,4.0,4.5")
    parser.add_argument("--flow", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=25)
    parser.add_argument("--buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-tr", type=float, default=6.0)
    parser.add_argument("--solvent-strength-slope", type=float, default=4.0)
    parser.add_argument("--acid-pka", type=float, default=4.2)
    parser.add_argument("--model", default="models/retention_time_catboost.cbm")
    parser.add_argument("--metadata", default="models/retention_time_catboost.json")
    args = parser.parse_args()

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    model = CatBoostRegressor()
    model.load_model(args.model)

    predictions = []
    for organic in numbers(args.organic):
        for ph in numbers(args.ph):
            row = {
                "smiles": args.smiles,
                "column_chemistry": args.column,
                "column_length_mm": args.length,
                "column_id_mm": args.column_id,
                "particle_um": args.particle,
                "modifier": args.modifier,
                "organic_percent": organic,
                "pH": ph,
                "buffer": args.buffer,
                "flow_ml_min": args.flow,
                "temperature_c": args.temperature,
                "condition_completeness": 1.0,
            }
            predicted_log = float(model.predict(np.asarray([featurize(row)], dtype=object))[0])
            predicted_tr = max(0.0, math.expm1(predicted_log))
            predictions.append({
                "organic_percent": organic,
                "pH": ph,
                "raw_ml_tR_min": predicted_tr,
            })

    # The sparse model supplies an absolute anchor. Relative changes across the
    # grid use transparent RP-HPLC physics until the dataset learns them itself.
    reference_organic = sorted(numbers(args.organic))[len(numbers(args.organic)) // 2]
    reference_ph = min(numbers(args.ph))
    anchor_candidates = [
        item["raw_ml_tR_min"] for item in predictions
        if item["organic_percent"] == reference_organic and item["pH"] == reference_ph
    ]
    anchor_tr = anchor_candidates[0]
    geometric_volume_ml = math.pi * (args.column_id / 2) ** 2 * args.length / 1000
    estimated_t0 = geometric_volume_ml * 0.68 / args.flow
    reference_k = max((anchor_tr - estimated_t0) / estimated_t0, 0.05)
    descriptors = compute_descriptors(args.smiles)
    is_carboxylic_acid = descriptors.get("CarboxylicAcidCount", 0) > 0

    def acid_retention_factor(ph: float) -> float:
        if not is_carboxylic_acid:
            return 1.0
        neutral_fraction = 1 / (1 + 10 ** (ph - args.acid_pka))
        return 0.15 + 0.85 * neutral_fraction

    reference_ion_factor = acid_retention_factor(reference_ph)
    for item in predictions:
        solvent_factor = 10 ** (
            -args.solvent_strength_slope * (item["organic_percent"] - reference_organic) / 100
        )
        ion_factor = acid_retention_factor(item["pH"]) / reference_ion_factor
        adjusted_k = reference_k * solvent_factor * ion_factor
        item["predicted_tR_min"] = estimated_t0 * (1 + adjusted_k)
        item["distance_to_target"] = abs(item["predicted_tR_min"] - args.target_tr)

    predictions.sort(key=lambda item: (item["distance_to_target"], item["organic_percent"], item["pH"]))
    print("Provisional direct-tR screening (experimental)")
    print(f"Training: {metadata['rows']} rows / {metadata['unique_molecules']} molecules")
    print(f"Grouped CV MAE: {metadata['group_cv_mae_min']:.2f} min")
    print(f"Target tR: {args.target_tr:.2f} min\n")
    for rank, item in enumerate(predictions, 1):
        print(
            f"#{rank:02d}  {args.modifier} {item['organic_percent']:g}%  "
            f"pH {item['pH']:g}  -> hybrid tR {item['predicted_tR_min']:.2f} min "
            f"(raw ML {item['raw_ml_tR_min']:.2f})"
        )
    spread = max(item["predicted_tR_min"] for item in predictions) - min(
        item["predicted_tR_min"] for item in predictions
    )
    print(f"\nPrediction spread across grid: {spread:.2f} min")
    resolution_threshold = max(0.5, 0.2 * float(metadata["group_cv_mae_min"]))
    if spread < resolution_threshold:
        print(
            f"Ranking status: UNRESOLVED — spread is below the {resolution_threshold:.2f} min "
            "minimum resolution implied by model error."
        )
    means_by_organic = {}
    for organic in sorted({item["organic_percent"] for item in predictions}):
        values = [item["predicted_tR_min"] for item in predictions if item["organic_percent"] == organic]
        means_by_organic[organic] = sum(values) / len(values)
    monotonic_rp = all(
        means_by_organic[left] >= means_by_organic[right]
        for left, right in zip(means_by_organic, list(means_by_organic)[1:])
    )
    print(f"RP-HPLC organic trend sanity: {'PASS' if monotonic_rp else 'FAIL'}")
    print(
        f"Physics assumptions: estimated t0={estimated_t0:.2f} min, "
        f"solvent-strength slope={args.solvent_strength_slope:g}"
        + (f", carboxylic-acid pKa={args.acid_pka:g}" if is_carboxylic_acid else "")
    )
    print("Reliability: LOW — hybrid values rank experiments; they are not validated absolute tR predictions.")


if __name__ == "__main__":
    main()
