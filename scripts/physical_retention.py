"""Calibrate an isocratic LSS model from reviewed measurements, never from ML labels.

This deliberately does not simulate gradients or infer peak widths.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def validate(rows):
    if not rows:
        raise ValueError("No measurements")
    contexts, identities, ids = set(), set(), set()
    required = ("column", "modifier", "aqueous_recipe", "pH", "temperature_c",
                "flow_ml_min", "instrument_id", "pH_convention")
    for r in rows:
        if r.get("reviewed") is not True or r.get("mode") != "isocratic":
            raise ValueError("Only reviewed isocratic measurements are accepted")
        if r.get("t0_origin") != "measured" or not r.get("t0_evidence") or not r.get("source"):
            raise ValueError("Measured t0 evidence and measurement source are required")
        if r.get("label_origin") != "experiment":
            raise ValueError("Predictions and synthetic labels are not calibration evidence")
        identity = r.get("full_inchikey")
        if not identity or len(identity.split("-")) != 3:
            raise ValueError("Full molecular identity is required")
        identities.add(identity)
        if not r.get("observation_id") or r["observation_id"] in ids:
            raise ValueError("Unique observation IDs are required")
        ids.add(r["observation_id"])
        context = r.get("context", {})
        if any(context.get(k) in (None, "") for k in required):
            raise ValueError("Complete fixed chromatographic context is required")
        for key in ("pH", "temperature_c", "flow_ml_min"):
            value = context[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid context {key}")
        if context["flow_ml_min"] <= 0 or context["temperature_c"] <= -273.15:
            raise ValueError("Nonphysical flow or temperature")
        contexts.add(json.dumps(context, sort_keys=True))
        for k in ("phi", "t0_min", "tr_min"):
            if isinstance(r.get(k), bool) or not isinstance(r.get(k), (float, int)) or not math.isfinite(r[k]):
                raise ValueError(f"Invalid {k}")
        if not 0 <= r["phi"] <= 1 or not 0 < r["t0_min"] < r["tr_min"]:
            raise ValueError("Require fraction phi in [0,1] and tr > measured t0 > 0")
        if r.get("split") not in ("calibration", "validation"):
            raise ValueError("Preassigned calibration/validation split is required")
    if len(contexts) != 1 or len(identities) != 1:
        raise ValueError("Fit one exact molecule and one fixed chromatographic context at a time")
    fit_phi = {r["phi"] for r in rows if r["split"] == "calibration"}
    test_phi = {r["phi"] for r in rows if r["split"] == "validation"}
    if len(fit_phi) < 3:
        raise ValueError("Need at least three distinct calibration compositions; repeats do not count")
    if fit_phi & test_phi:
        raise ValueError("Validation must hold out complete compositions, including replicates")


def calibrate(rows):
    validate(rows)
    fit = [r for r in rows if r["split"] == "calibration"]
    # Equal weight per composition, not per number of replicate injections.
    phis = sorted({r["phi"] for r in fit})
    y = [np.mean([math.log10((r["tr_min"]-r["t0_min"])/r["t0_min"])
                  for r in fit if r["phi"] == phi]) for phi in phis]
    slope, intercept = np.polyfit(phis, y, 1)
    if slope >= 0:
        raise ValueError("Data do not support decreasing retention with organic fraction under LSS")
    model = {"log10_kw": float(intercept), "S": float(-slope),
             "phi_domain": [min(phis), max(phis)], "context": rows[0]["context"],
             "full_inchikey": rows[0]["full_inchikey"],
             "calibration_ids": [r["observation_id"] for r in fit],
             "validation": [], "predicted_Rs": None, "gradient_supported": False,
             "separation_confirmed": False, "status": "calibrated_not_independently_validated"}
    for r in rows:
        if r["split"] == "validation":
            inside = min(phis) <= r["phi"] <= max(phis)
            pred = r["t0_min"]*(1+10**(intercept+slope*r["phi"])) if inside else None
            model["validation"].append({"observation_id": r["observation_id"],
                "observed_tr_min": r["tr_min"], "predicted_tr_min": pred,
                "absolute_error_min": abs(pred-r["tr_min"]) if inside else None,
                "status": "in_domain" if inside else "outside_calibrated_compositions"})
    errors = [r["absolute_error_min"] for r in model["validation"] if r["absolute_error_min"] is not None]
    model["validation_mae_min"] = float(np.mean(errors)) if errors else None
    if errors:
        model["status"] = "held_out_compositions_evaluated_no_accuracy_acceptance_claim"
    model["evidence"] = rows
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, required=True, help="JSON array; explicit measured t0 and fixed context")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = calibrate(json.loads(args.measurements.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(result["status"])


if __name__ == "__main__":
    main()
