"""Research LSS gradient retention: real measurements -> calibration -> simulation.

Ideal equilibrium, dilute load, constant flow/temperature, one modifier, no solvent
adsorption/dispersion. Corrected-time migration integral:
integral_0^(tr-t0-extra) 1/k(phi(t-dwell)) dt = t0.
No peak widths, Rs, detector response or automatic training-data promotion.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def finite(value, lower=0, inclusive=True):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value >= lower if inclusive else value > lower))


def validate_system(system):
    if system.get("time_reference") != "sample_entry_at_column_inlet":
        raise ValueError("Align sample/program time origin to column inlet; extra time is post-column only")
    for key in ("t0_column_min", "dwell_time_min", "extra_column_time_min"):
        if not finite(system.get(key), inclusive=key != "t0_column_min"):
            raise ValueError(f"Measured timing required: {key}")
        if not system.get(key+"_evidence"):
            raise ValueError(f"Missing timing evidence: {key}")
    context = system.get("context", {})
    for key in ("column", "modifier", "phase_recipes", "pH_convention", "instrument_id"):
        if not context.get(key):
            raise ValueError(f"Missing fixed context: {key}")
    for key in ("temperature_c", "pH", "flow_ml_min"):
        if not finite(context.get(key), lower=-273.15 if key == "temperature_c" else 0,
                      inclusive=key != "flow_ml_min"):
            raise ValueError(f"Missing/invalid context: {key}")
    if system.get("reviewed") is not True:
        raise ValueError("System timings, context and approximation applicability need review")


def validate_program(program):
    if len(program) < 2:
        raise ValueError("Complete time/fraction program required")
    for point in program:
        if not finite(point.get("time_min")) or not finite(point.get("phi")) or point["phi"] > 1:
            raise ValueError("Time must be finite/nonnegative; phi is a fraction in [0,1]")
    if program[0]["time_min"] != 0 or any(b["time_min"] <= a["time_min"] for a,b in zip(program,program[1:])):
        raise ValueError("Program must start at zero and strictly increase in time")


def integral(tau, logkw, strength, program, dwell):
    """Analytic integration on piecewise linear solvent segments, including delay.

Private numerical kernel: callers validate the program and bound integration time.
Initial mobile phase exists before the delayed program reaches the inlet.
"""
    knots = sorted({0., tau, *[p["time_min"]+dwell for p in program
                              if 0 < p["time_min"]+dwell < tau]})
    times = [p["time_min"] for p in program]
    fractions = [p["phi"] for p in program]
    total = 0.
    for left,right in zip(knots,knots[1:]):
        a,b = np.interp([left-dwell,right-dwell],times,fractions)
        span = right-left
        exponent = math.log(10)*strength*(b-a)
        factor = math.expm1(exponent)/exponent if abs(exponent) > 1e-10 else 1.
        total += 10**(-logkw+strength*a)*span*factor
    return float(total)


def retention(logkw, strength, program, system):
    validate_program(program)
    if not finite(logkw, -5) or logkw > 15 or not finite(strength, 0, False) or strength > 50:
        raise ValueError("LSS parameters outside numerical research bounds")
    t0,extra,dwell = (system[k] for k in ("t0_column_min","extra_column_time_min","dwell_time_min"))
    horizon = program[-1]["time_min"]-t0-extra
    if horizon <= 0 or integral(horizon,logkw,strength,program,dwell) < t0:
        return None  # No invented elution after the supplied acquisition window.
    tau = brentq(lambda t: integral(t,logkw,strength,program,dwell)-t0,0,horizon,xtol=1e-10)
    return float(tau+t0+extra)


def fit_campaign(campaign):
    system = campaign["system"]
    validate_system(system)
    identity = campaign.get("full_inchikey", "")
    if len(identity.split("-")) != 3:
        raise ValueError("Full InChIKey required; fit each molecule separately")
    rows = campaign.get("observations", [])
    ids, split_programs = set(), {"calibration": set(), "validation": set()}
    for row in rows:
        validate_program(row["program"])
        if (row.get("label_origin") != "experiment" or row.get("reviewed") is not True
                or not row.get("source") or not row.get("observation_id")):
            raise ValueError("Only reviewed experimental labels with source and ID are accepted")
        if row["observation_id"] in ids:
            raise ValueError("Duplicate observation ID")
        ids.add(row["observation_id"])
        if row.get("split") not in split_programs:
            raise ValueError("Preassigned calibration/validation split required")
        if (not finite(row.get("tr_min"), system["t0_column_min"]+system["extra_column_time_min"], False)
                or row["tr_min"] > row["program"][-1]["time_min"]):
            raise ValueError("Require measured retention after void time and inside acquisition window")
        split_programs[row["split"]].add(digest(row["program"]))
    if len(split_programs["calibration"]) < 3:
        raise ValueError("Need at least three different calibration programs; replicates do not count")
    if split_programs["calibration"] & split_programs["validation"]:
        raise ValueError("Hold out complete programs, including their replicates")
    fit = [r for r in rows if r["split"] == "calibration"]
    # Equal total weight per program regardless of replicate count.
    multiplicity = {key: sum(digest(r["program"]) == key for r in fit)
                    for key in split_programs["calibration"]}
    def residual(params):
        return [math.log(integral(r["tr_min"]-system["t0_column_min"]-system["extra_column_time_min"],
                                 *params,r["program"],system["dwell_time_min"])/system["t0_column_min"])
                / math.sqrt(multiplicity[digest(r["program"])]) for r in fit]
    solutions = [least_squares(residual, start, bounds=([-5,.01],[15,50]))
                 for start in ([2,3],[5,10],[1,1])]
    best = min(solutions,key=lambda s:s.cost)
    condition = float(np.linalg.cond(best.jac))
    at_bound = any(abs(v-lo)<1e-4 or abs(v-hi)<1e-4
                   for v,lo,hi in zip(best.x,[-5,.01],[15,50]))
    if not best.success or not math.isfinite(condition) or condition > 1e6 or at_bound:
        raise ValueError("LSS parameters not reliably identifiable; need more informative real programs")
    result = {"schema_version": 1, "model_kind": "ideal_gradient_lss", "full_inchikey": identity,
              "system": deepcopy(system), "log10_kw": float(best.x[0]), "S": float(best.x[1]),
              "fit_jacobian_condition_number": condition, "campaign_sha256": digest(campaign),
              "calibration_ids": [r["observation_id"] for r in fit],
              "calibration_programs": [r["program"] for r in fit],
              "status": "research_calibration_not_validated_for_training_augmentation",
              "validation": [], "training_augmentation_approved": False,
              "evidence": deepcopy(rows)}
    for row in rows:
        if row["split"] == "validation":
            prediction = retention(*best.x,row["program"],system)
            result["validation"].append({"observation_id": row["observation_id"],
                "observed_tr_min": row["tr_min"], "predicted_tr_min": prediction,
                "absolute_error_min": abs(prediction-row["tr_min"]) if prediction is not None else None})
    errors = [r["absolute_error_min"] for r in result["validation"] if r["absolute_error_min"] is not None]
    result["validation_mae_min"] = float(np.mean(errors)) if errors else None
    result["validation_nonelution_count"] = sum(r["predicted_tr_min"] is None for r in result["validation"])
    return result


def simulate(model, program):
    validate_system(model["system"])
    validate_program(program)
    # Recompute from the evidence so edited parameters cannot masquerade as calibration.
    campaign = {"system": model["system"], "full_inchikey": model["full_inchikey"], "observations": model["evidence"]}
    verified = fit_campaign(campaign)
    if not np.allclose([model["log10_kw"], model["S"]], [verified["log10_kw"],verified["S"]],rtol=1e-8,atol=1e-8):
        raise ValueError("Model parameters disagree with calibration evidence")
    known = {digest(p) for p in verified["calibration_programs"]}
    prediction = retention(model["log10_kw"],model["S"],program,model["system"])
    return {"label_origin": "simulation", "full_inchikey": model["full_inchikey"],
            "model_sha256": digest(model), "program": program,
            "predicted_tr_min": prediction, "predicted_Rs": None, "peak_width_min": None,
            "status": "simulated_elution" if prediction is not None else "not_eluted_within_program",
            "domain": "calibration_program" if digest(program) in known else "new_program_unvalidated",
            "training_ready": False, "separation_confirmed": False,
            "warning": "Synthetic research output, not an experimental measurement or an accuracy guarantee."}


def selected_template(recommendation):
    c = recommendation["primary"]["candidate"]
    m = c["conditions"]
    # Initial adapter is deliberately restricted to binary water / pure modifier.
    if float(m["eluent.A.h2o"]) != 100 or float(m["eluent.B.meoh"]) != 100:
        raise ValueError("Template adapter supports only water A / methanol B")
    for p in c["gradient_program"]:
        if float(p["C [%]"]) or float(p["D [%]"]) or float(p["flow rate [ml/min]"]) != float(m["column.flowrate"]):
            raise ValueError("Requires binary composition and constant flow")
    program = [{"time_min": float(p["t [min]"]), "phi": float(p["B [%]"])/100} for p in c["gradient_program"]]
    validate_program(program)
    system = {"reviewed": False, "time_reference": None, "context": {
        "column": {k:v for k,v in m.items() if k in ("column.name","column.length","column.id","column.particle.size")},
        "modifier": "methanol", "phase_recipes": {k:v for k,v in m.items() if k.startswith(("eluent.A.","eluent.B."))},
        "pH": float(m["eluent.A.pH"]), "pH_convention": None,
        "temperature_c": float(m["column.temperature"]), "flow_ml_min": float(m["column.flowrate"]),
        "instrument_id": None}}
    for key in ("t0_column_min", "dwell_time_min", "extra_column_time_min"):
        system[key] = None
        system[key+"_evidence"] = None
    return {"status": "WAITING_FOR_MEASURED_TIMINGS_AND_CALIBRATION_SERIES",
            "source_candidate_id": c["candidate_id"], "source_gradient_sha256": digest(c["gradient_program"]),
            "selected_program": program, "original_gradient_program": c["gradient_program"],
            "system": system, "full_inchikey": None, "observations": [],
            "observation_template": {"observation_id": None, "program": program, "tr_min": None,
                "split": None, "source": None, "label_origin": "experiment", "reviewed": False},
            "note": "One campaign per exact molecule. Preserve selected gradient. Add actual calibration/held-out programs; CatBoost predictions are not measurements."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["template","fit","simulate"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--program", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    if args.command == "template":
        result = selected_template(data)
    elif args.command == "fit":
        result = fit_campaign(data)
    else:
        if args.program is None:
            p.error("--program is required for simulation")
        result = simulate(data,json.loads(args.program.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite fitted models or research predictions.
    with args.out.open("x") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write("\n")
    print(result["status"])


if __name__ == "__main__":
    main()
