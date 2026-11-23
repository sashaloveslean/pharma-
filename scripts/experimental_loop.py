#!/usr/bin/env python3
"""Small-data TFV/FTC loop: frozen plan -> real observations -> local bias -> next test.

No global retraining, invented measurements, Rs-from-two-RTs, or automatic execution.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from curate_target_evidence import FULL_KEYS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def positive(value, zero=False):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def freeze_predictions(folder, protocol, comparison, predictor=None):
    """Freeze once before measurements; never rewrite an existing scientific baseline."""
    path = folder / "predictions.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("status") == "FROZEN_BASE_PREDICTIONS":
            if previous.get("protocol_sha256") != digest(protocol) or previous.get("model_sha256") != comparison["model_sha256"]:
                raise ValueError("Frozen protocol/model changed. Create a new campaign folder; previous predictions are preserved.")
            return previous
    log = folder / "observations.jsonl"
    if log.exists() and log.read_text().strip():
        raise ValueError("Cannot establish predictions retrospectively after observations were recorded")
    predictions = (predictor or predict_base)(protocol, comparison)
    if predictions.get("status") == "FROZEN_BASE_PREDICTIONS":
        # Exclusive creation: an existing blocked-status file is not a scientific baseline.
        if path.exists():
            diagnostic = folder / "blocked_predictions.json"
            if diagnostic.exists():
                raise ValueError("Existing blocked predictions diagnostic; preserve it and use a new campaign folder")
            path.rename(diagnostic)
        with path.open("x") as file:
            json.dump(predictions, file, ensure_ascii=False, indent=2)
    else:
        (folder / "prediction_blockers.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    return predictions


def import_observation(folder, record):
    """Validate a real injection before appending it; no silent rewriting of observations."""
    protocol = json.loads((folder / "protocol.json").read_text())
    frozen = json.loads((folder / "predictions.json").read_text())
    if protocol_issues(protocol) or frozen.get("protocol_sha256") != digest(protocol):
        raise ValueError("Approved unchanged protocol and prospective frozen predictions are required")
    log = folder / "observations.jsonl"
    existing = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    _, rejected = validate_observations(protocol, frozen, existing + [record])
    if rejected:
        raise ValueError("Observation log validation failed: " + json.dumps(rejected, ensure_ascii=False))
    with log.open("a") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def scaled_ramp(program, scale):
    """Vary analytical gradient only; retain source wash/return/equilibration durations."""
    result = deepcopy(program)
    maximum = max(float(p["B [%]"]) for p in program)
    boundary = next(float(p["t [min]"]) for p in program if float(p["B [%]"]) == maximum)
    for point in result:
        t = float(point["t [min]"])
        point["t [min]"] = round(t*scale if t <= boundary else t + boundary*(scale-1), 6)
    return result


def initialize(recommendation, out):
    if not recommendation.get("primary"):
        raise ValueError("A primary source method is required")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        raise ValueError("protocol.json already exists; laboratory-owned protocol will not be overwritten")
    c = recommendation["primary"]["candidate"]
    protocol = {"schema_version": 1, "approval_status": "draft", "analyst": None,
                "target_inchikeys": FULL_KEYS, "source_candidate": deepcopy(c),
                "instrument": {"instrument_id": None, "column_serial": None, "matrix_id": None,
                               "detection_protocol_id": None, "dwell_volume_ml": None, "dwell_measurement_method": None,
                               "pressure_limit_bar": None},
                "equilibration_confirmed": False,
                "sample": {"injection_ul": None, "tfv_concentration_ug_ml": None, "ftc_concentration_ug_ml": None,
                           "diluent_recipe": None},
                "recipes": [{"recipe_id": "pH_low", "proposed_pH": 2.6, "metadata": None,
                             "recipe_family_id": None, "ph_measurement_convention": None},
                            {"recipe_id": "pH_high", "proposed_pH": 3.6, "metadata": None,
                             "recipe_family_id": None, "ph_measurement_convention": None}],
                "acceptance": {"target_rs": None, "maximum_tailing": None, "minimum_snr": None,
                               "maximum_carryover_percent": None, "minimum_k": None},
                "experiments": []}
    for recipe in protocol["recipes"]:
        for scale in (.8, 1., 1.2):
            protocol["experiments"].append({"experiment_id": f"{recipe['recipe_id']}_ramp_{scale}",
                "recipe_id": recipe["recipe_id"], "role": "adaptation", "ramp_scale": scale,
                "gradient_program": scaled_ramp(c["gradient_program"], scale)})
    for recipe, scale in zip(protocol["recipes"], (.9, 1.1)):
        protocol["experiments"].append({"experiment_id": f"reserved_{recipe['recipe_id']}_ramp_{scale}",
            "recipe_id": recipe["recipe_id"], "role": "validation", "ramp_scale": scale,
            "gradient_program": scaled_ramp(c["gradient_program"], scale)})
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    template = {"run_id": None, "experiment_id": None, "sample_type": "mixture", "review_status": "unreviewed", "actual_method": None,
                "raw_signal_path": None, "raw_signal_sha256": None, "raw_signal_record_id": None,
                "pressure_bar": None, "t0_min": None, "t0_measurement_method": None,
                "reported_rs": None, "reported_rs_method": None, "carryover_percent": None,
                "peaks": {name: {"inchikey": key, "status": "unidentified", "tr_min": None,
                                  "width_half_height_min": None, "tailing": None, "snr": None}
                          for name, key in FULL_KEYS.items()},
                "notes": "Statuses: identified, coeluted, not_detected, unidentified. Never invent tR for failed runs."}
    (out / "observation_template.json").write_text(json.dumps(template, indent=2, ensure_ascii=False))
    (out / "phase_recipe_template.json").write_text(json.dumps({
        "origin": "source recipe, not automatically valid for both proposed pH levels",
        "metadata": {k:v for k,v in c["conditions"].items() if k.startswith("eluent.")}}, indent=2))
    if not (out / "observations.jsonl").exists():
        (out / "observations.jsonl").touch()
    return protocol


def protocol_issues(protocol):
    from compare_columns import method_issues
    from recommend_lab import cycle_issues
    issues = []
    if protocol.get("approval_status") != "approved" or not protocol.get("analyst"):
        issues.append("analyst_approval_required")
    if protocol.get("target_inchikeys") != FULL_KEYS:
        issues.append("target_identity_changed")
    instrument = protocol.get("instrument", {})
    for key in ("instrument_id", "column_serial", "matrix_id", "detection_protocol_id", "dwell_measurement_method"):
        if not instrument.get(key): issues.append("missing_"+key)
    if not positive(instrument.get("dwell_volume_ml"), zero=True): issues.append("missing_measured_dwell_volume")
    if not positive(instrument.get("pressure_limit_bar")): issues.append("missing_pressure_limit")
    for key in ("injection_ul", "tfv_concentration_ug_ml", "ftc_concentration_ug_ml"):
        if not positive(protocol.get("sample", {}).get(key)): issues.append("missing_fixed_sample_"+key)
    if not protocol.get("sample", {}).get("diluent_recipe"): issues.append("missing_diluent_recipe")
    for key,value in protocol.get("acceptance", {}).items():
        if value is not None and not positive(value, zero=key in {"minimum_k", "maximum_carryover_percent"}):
            issues.append("invalid_acceptance_criterion:"+key)
    if not protocol.get("equilibration_confirmed"): issues.append("equilibration_not_confirmed")
    recipes = protocol.get("recipes", [])
    ids = [r["recipe_id"] for r in recipes]
    if len(set(ids)) != len(ids): issues.append("duplicate_recipe_ids")
    families = {r.get("recipe_family_id") for r in recipes}
    if len(families) != 1 or not all(families): issues.append("common_recipe_family_not_confirmed")
    for r in recipes:
        if not r.get("metadata") or not r.get("ph_measurement_convention"):
            issues.append("recipe_preparation_or_ph_convention_missing:"+r["recipe_id"])
        for key,value in (r.get("metadata") or {}).items():
            parts = key.split(".")
            if len(parts) != 3 or parts[0] != "eluent" or parts[2] in {
                "pH", "h2o", "meoh", "acn", "iproh", "acetone", "hex", "chcl3", "ch2cl2", "hept"}:
                continue
            if value not in ("", "0", 0, None) and not r["metadata"].get(key+".unit"):
                issues.append("additive_concentration_unit_missing:"+key)
    # Different solvent mixtures cannot be passed off as a pH-only factor.
    if recipes and all(r.get("metadata") for r in recipes):
        solvents = ("h2o", "meoh", "acn", "iproh", "acetone", "hex", "chcl3", "ch2cl2", "hept")
        signature = lambda r: [float(r["metadata"].get(f"eluent.{c}.{s}") or 0) for c in "ABCD" for s in solvents]
        if any(signature(r) != signature(recipes[0]) for r in recipes[1:]):
            issues.append("solvent_composition_changes_between_ph_levels")
        buffers = ("nh4form", "nh4ac", "nh4carb", "nh4bicarb", "nh4f", "phosphor")
        buffer_signature = lambda r: [(str(r["metadata"].get(f"eluent.{c}.{s}") or "0"),
                                       r["metadata"].get(f"eluent.{c}.{s}.unit") or "") for c in "ABCD" for s in buffers]
        if any(buffer_signature(r) != buffer_signature(recipes[0]) for r in recipes[1:]):
            issues.append("buffer_species_or_concentration_changes_between_ph_levels")
        ph_values = [float(r["metadata"].get("eluent.A.pH") or 0) for r in recipes]
        if len(set(ph_values)) != len(ph_values): issues.append("ph_levels_are_not_distinct")
    experiments = protocol.get("experiments", [])
    if len({e["experiment_id"] for e in experiments}) != len(experiments): issues.append("duplicate_experiment_ids")
    for e in experiments:
        if e.get("recipe_id") not in ids or e.get("role") not in {"adaptation", "validation"}:
            issues.append("unknown_recipe_or_role:"+e["experiment_id"]); continue
        if not positive(e.get("ramp_scale")):
            issues.append("invalid_ramp_scale:"+e["experiment_id"])
        recipe = next(r for r in recipes if r["recipe_id"] == e["recipe_id"])
        if not recipe.get("metadata"): continue
        method = actual_method(protocol, e)
        if e["gradient_program"] != scaled_ramp(protocol["source_candidate"]["gradient_program"], e["ramp_scale"]):
            issues.append("unplanned_gradient_or_flow_change:"+e["experiment_id"])
        issues.extend(f"{e['experiment_id']}:{p}" for p in method_issues({
            "metadata": method["metadata"], "gradient_program": method["gradient_program"],
            "source_info": protocol["source_candidate"]["method_source"]}))
        issues.extend(f"{e['experiment_id']}:{p}" for p in cycle_issues(method["gradient_program"]))
    # A replicate or alternate identifier must not leak the same condition into validation.
    if not any("recipe_preparation" in issue for issue in issues):
        roles, condition_ids = defaultdict(set), defaultdict(set)
        for e in experiments:
            if e.get("recipe_id") in ids:
                signature = digest(actual_method(protocol, e))
                roles[signature].add(e["role"])
                condition_ids[signature].add(e["experiment_id"])
        if any(len(v)>1 for v in roles.values()): issues.append("same_method_in_adaptation_and_validation")
        if any(len(v)>1 for v in condition_ids.values()): issues.append("duplicate_physical_method_with_different_experiment_ids")
    return sorted(set(issues))


def actual_method(protocol, experiment):
    recipe = next(r for r in protocol["recipes"] if r["recipe_id"] == experiment["recipe_id"])
    metadata = deepcopy(protocol["source_candidate"]["conditions"])
    metadata = {k:v for k,v in metadata.items() if not k.startswith("eluent.") and k != "column.t0"}
    metadata.update(recipe["metadata"] or {})
    if any(not k.startswith("eluent.") for k in (recipe["metadata"] or {})):
        raise ValueError("Recipes may specify eluents only; fixed column/temperature/flow cannot be replaced")
    return deepcopy({"metadata": metadata, "gradient_program": experiment["gradient_program"],
            "instrument": protocol["instrument"], "sample": protocol["sample"],
            "ph_measurement_convention": recipe["ph_measurement_convention"]})


def predict_base(protocol, comparison):
    from catboost import CatBoostRegressor
    from realistic_features import ChargeCache, enriched_vector
    from train_open_retention import vector
    issues = protocol_issues(protocol)
    if issues: return {"status": "BLOCKED_PROTOCOL", "issues": issues, "predictions": {}}
    model_path = Path(comparison["selected_evaluation"]).parent / (comparison["selected_model"]+".cbm")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != comparison["model_sha256"]:
        raise ValueError("Base model checksum changed")
    evaluation = json.loads(Path(comparison["selected_evaluation"]).read_text())
    demo = json.loads(Path("reports/tenofovir_demo/result.json").read_text())
    smiles = {name: info["payload"]["PropertyTable"]["Properties"][0]["SMILES"] for name,info in demo["pubchem"].items()}
    model = CatBoostRegressor(); model.load_model(str(model_path))
    cache = ChargeCache(Path("data/processed/public_retention/research_cache/charges.json"))
    predictions = {}
    for experiment in protocol["experiments"]:
        method = actual_method(protocol, experiment)
        predictions[experiment["experiment_id"]] = {"method_sha256": digest(method), "actual_method_template": method, "tr_min": {}}
        for name, structure in smiles.items():
            row = {**method, "smiles": structure}
            features = vector(row, evaluation["numeric_metadata_features"]) if comparison["selected_model"] == "baseline" else enriched_vector(
                row, evaluation["numeric_metadata_features"], cache, evaluation["feature_version"])
            predictions[experiment["experiment_id"]]["tr_min"][name] = max(0.,float(np.expm1(model.predict([features])[0])))
    cache.save()
    return {"status": "FROZEN_BASE_PREDICTIONS", "protocol_sha256": digest(protocol), "model_sha256": comparison["model_sha256"],
            "dwell_handling": "recorded and exact-context gated; base model has no learned dwell correction",
            "predictions": predictions, "calibrated_for_this_instrument": False}


def validate_observations(protocol, frozen, records):
    accepted, rejected, seen_runs, seen_signals = [], [], set(), set()
    experiments = {e["experiment_id"]: e for e in protocol["experiments"]}
    for index, r in enumerate(records):
        issues = []
        if r.get("label_origin", "experiment") != "experiment":
            issues.append("synthetic_or_predicted_label_is_not_a_lab_measurement")
        eid, rid = r.get("experiment_id"), r.get("run_id")
        if not rid or rid in seen_runs: issues.append("missing_or_duplicate_run_id")
        seen_runs.add(rid)
        if eid not in experiments: issues.append("unknown_experiment")
        if r.get("review_status") != "approved": issues.append("unreviewed_observation")
        if r.get("sample_type") not in {"mixture", "blank", "standard_tenofovir", "standard_emtricitabine"}:
            issues.append("unknown_sample_type")
        expected = frozen.get("predictions", {}).get(eid)
        if not expected or digest(r.get("actual_method")) != expected["method_sha256"]:
            issues.append("actual_method_or_instrument_differs_from_frozen_prediction")
        signal = Path(r.get("raw_signal_path") or "__missing_signal__")
        if not signal.is_file() or file_digest(signal) != r.get("raw_signal_sha256"):
            issues.append("raw_signal_missing_or_checksum_mismatch")
        signal_id = (r.get("raw_signal_sha256"), r.get("raw_signal_record_id"))
        if not signal_id[1] or signal_id in seen_signals: issues.append("missing_or_duplicate_raw_injection")
        seen_signals.add(signal_id)
        if not positive(r.get("pressure_bar")): issues.append("missing_measured_pressure")
        for name,key in FULL_KEYS.items():
            peak = r.get("peaks", {}).get(name, {})
            if peak.get("inchikey") != key: issues.append("target_identity_mismatch:"+name)
            if peak.get("status") not in {"identified", "coeluted", "not_detected", "unidentified"}:
                issues.append("invalid_peak_status:"+name)
            if peak.get("status") == "identified" and not positive(peak.get("tr_min")):
                issues.append("identified_peak_needs_exact_tr:"+name)
            if peak.get("status") != "identified" and (peak.get("tr_min") is not None or peak.get("width_half_height_min") is not None):
                issues.append("failed_or_unidentified_peak_cannot_supply_exact_label:"+name)
            for field in ("width_half_height_min", "tailing", "snr"):
                if peak.get(field) is not None and not positive(peak[field]): issues.append("invalid_"+field+":"+name)
        if r.get("t0_min") is not None and (not positive(r["t0_min"]) or not r.get("t0_measurement_method")):
            issues.append("invalid_t0_or_missing_measurement_method")
        if r.get("reported_rs") is not None and not r.get("reported_rs_method"):
            issues.append("reported_rs_method_required")
        for field in ("reported_rs", "carryover_percent"):
            if r.get(field) is not None and not positive(r[field], zero=True): issues.append("invalid_"+field)
        if issues:
            rejected.append({"row": index+1, "run_id": rid, "issues": sorted(set(issues))})
        else:
            accepted.append({**r, "role": experiments[eid]["role"], "recipe_id": experiments[eid]["recipe_id"]})
    return accepted, rejected


def measured_rs(row):
    if row.get("sample_type") != "mixture": return None
    peaks = row["peaks"]
    if not all(p["status"] == "identified" for p in peaks.values()): return None
    if row.get("reported_rs") is not None: return row["reported_rs"]
    a,b = [peaks[name] for name in FULL_KEYS]
    if all(positive(p.get("width_half_height_min")) for p in (a,b)):
        return 1.18*abs(a["tr_min"]-b["tr_min"])/(a["width_half_height_min"]+b["width_half_height_min"])
    return None


def suitability_issues(row, protocol):
    issues = []
    criteria = protocol.get("acceptance", {})
    for key in ("target_rs", "maximum_tailing", "minimum_snr", "maximum_carryover_percent", "minimum_k"):
        threshold = criteria.get(key)
        if threshold is None:
            issues.append("criterion_unset:"+key); continue
        if key == "target_rs":
            value = measured_rs(row)
            if value is None or value < threshold: issues.append("insufficient_or_unknown_rs")
        elif key == "maximum_carryover_percent":
            if row.get("carryover_percent") is None or row["carryover_percent"] > threshold:
                issues.append("excessive_or_unknown_carryover")
        elif key == "minimum_k":
            if not row.get("t0_min") or any(p["status"] != "identified" or (p["tr_min"]-row["t0_min"])/row["t0_min"] < threshold
                                         for p in row["peaks"].values()): issues.append("insufficient_or_unknown_retention")
        else:
            field = "tailing" if key == "maximum_tailing" else "snr"
            if any(p.get(field) is None or (p[field] > threshold if field == "tailing" else p[field] < threshold)
                   for p in row["peaks"].values()): issues.append("failed_or_unknown_"+field)
    return sorted(set(issues))


def local_bias(protocol, frozen, observations):
    fits = {}
    for recipe in protocol["recipes"]:
        for name in FULL_KEYS:
            grouped = defaultdict(list)
            for row in observations:
                peak = row["peaks"][name]
                if row.get("sample_type") == "mixture" and row["role"] == "adaptation" and row["recipe_id"] == recipe["recipe_id"] and peak["status"] == "identified":
                    grouped[row["experiment_id"]].append(peak["tr_min"]-frozen["predictions"][row["experiment_id"]]["tr_min"][name])
            errors = [float(np.mean(v)) for v in grouped.values()]
            if len(errors) < 3: continue
            baseline_mae = float(np.mean(np.abs(errors)))
            cv_mae = float(np.mean([abs(error-np.median(errors[:i]+errors[i+1:])) for i,error in enumerate(errors)]))
            use = cv_mae < baseline_mae
            fits[recipe["recipe_id"]+":"+name] = {"bias_min": float(np.median(errors)) if use else 0.,
                "used": bool(use), "unique_training_conditions": len(errors), "condition_ids": sorted(grouped),
                "leave_condition_out_mae_min": cv_mae, "base_condition_mae_min": baseline_mae,
                "ramp_scale_range": [min(e["ramp_scale"] for e in protocol["experiments"] if e["experiment_id"] in grouped),
                                     max(e["ramp_scale"] for e in protocol["experiments"] if e["experiment_id"] in grouped)],
                "interval_90_min": None, "status": "LOCAL_BIAS_PROVISIONAL_NOT_CALIBRATED"}
    return fits


def summarize(protocol, frozen, records):
    issues = protocol_issues(protocol)
    if issues: return {"status": "WAITING_FOR_APPROVED_PROTOCOL", "issues": issues, "local_model_fitted": False,
                       "instrument_execution_authorized": False}
    if frozen.get("status") != "FROZEN_BASE_PREDICTIONS" or frozen.get("protocol_sha256") != digest(protocol):
        return {"status": "FROZEN_PREDICTIONS_REQUIRED", "local_model_fitted": False, "instrument_execution_authorized": False}
    observations, rejected = validate_observations(protocol, frozen, records)
    fits = local_bias(protocol, frozen, observations)
    condition_predictions = {}
    for e in protocol["experiments"]:
        estimates = {}
        for name in FULL_KEYS:
            base = frozen["predictions"][e["experiment_id"]]["tr_min"][name]
            fit = fits.get(e["recipe_id"]+":"+name)
            in_domain = fit and fit["ramp_scale_range"][0] <= e["ramp_scale"] <= fit["ramp_scale_range"][1]
            estimates[name] = {"base_tr_min": base, "corrected_tr_min": max(0., base+fit["bias_min"]) if in_domain else None,
                               "local_correction_used": bool(in_domain and fit["used"]), "interval_90_min": None}
        condition_predictions[e["experiment_id"]] = estimates
    usable, validation, by_condition = [], [], defaultdict(list)
    for row in observations:
        if row.get("sample_type") != "mixture": continue
        eid = row["experiment_id"]; by_condition[eid].append(row)
        if row["role"] == "validation":
            for name in FULL_KEYS:
                peak = row["peaks"][name]; key = row["recipe_id"]+":"+name
                if peak["status"] != "identified" or key not in fits: continue
                base = frozen["predictions"][eid]["tr_min"][name]
                validation.append({"experiment_id": eid, "compound": name, "measured_min": peak["tr_min"],
                                   "base_absolute_error_min": abs(peak["tr_min"]-base),
                                   "corrected_absolute_error_min": abs(peak["tr_min"]-base-fits[key]["bias_min"])})
    for eid, rows in by_condition.items():
        values = [measured_rs(r) for r in rows]
        safe = all(r["pressure_bar"] <= protocol["instrument"]["pressure_limit_bar"] for r in rows)
        retained = not any(r.get("t0_min") and any(p["status"] == "identified" and p["tr_min"] <= r["t0_min"]
                                                  for p in r["peaks"].values()) for r in rows)
        if safe and retained and all(value is not None for value in values):
            usable.append({"experiment_id": eid, "min_measured_rs": min(values), "replicates": len(rows),
                           "suitability_issues": sorted({issue for row in rows for issue in suitability_issues(row, protocol)}),
                           "role": rows[0]["role"], "criterion": "minimum observed Rs across repeats; not a predicted Rs"})
    # Validation is never used to choose the next experiment or a best adaptation condition.
    adaptation = [r for r in usable if r["role"] == "adaptation"]
    best = max(adaptation, key=lambda r: (-sum(not issue.startswith("criterion_unset:") for issue in r["suitability_issues"]),
                                         r["min_measured_rs"], r["replicates"])) if adaptation else None
    todo = [e for e in protocol["experiments"] if e["role"] == "adaptation" and e["experiment_id"] not in by_condition]
    if todo:
        # Fill the designed grid, not a falsely precise ML-optimal Rs.
        recipe_order = {r["recipe_id"]: i for i,r in enumerate(protocol["recipes"])}
        e = min(todo, key=lambda e: (abs(e["ramp_scale"]-1), recipe_order[e["recipe_id"]], e["experiment_id"]))
        next_test = {"experiment_id": e["experiment_id"], "reason": "collect missing planned condition; not predicted separation optimum"}
    elif best and best["replicates"] < 2:
        next_test = {"experiment_id": best["experiment_id"], "reason": "repeat best observed adaptation condition"}
    elif best:
        pending = [e for e in protocol["experiments"] if e["role"] == "validation" and e["experiment_id"] not in by_condition]
        next_test = {"experiment_id": pending[0]["experiment_id"], "reason": "reserved validation; never used to fit or select"} if pending else None
    else:
        next_test = None
    failures = [r["run_id"] for r in observations if r.get("sample_type") == "mixture" and any(p["status"] != "identified" for p in r["peaks"].values())]
    front = [r["run_id"] for r in observations if r.get("t0_min") and any(
        p["status"] == "identified" and p["tr_min"] <= r["t0_min"] for p in r["peaks"].values())]
    action_required = None
    if front and not usable:
        next_test = None
        action_required = "REVIEW_RETENTION_BEFORE_OPTIMIZING_SEPARATION"
    if any(r["pressure_bar"] > protocol["instrument"]["pressure_limit_bar"] for r in observations):
        next_test = None
        action_required = "REVIEW_PRESSURE_LIMIT_VIOLATION_BEFORE_FURTHER_RUNS"
    return {"status": "LOCAL_CYCLE_PROVISIONAL" if observations else "WAITING_FOR_MEASUREMENTS",
            "accepted_runs": len(observations), "rejected": rejected, "failed_runs_preserved": failures,
            "at_or_before_measured_front": front, "local_model_fitted": any(f["used"] for f in fits.values()), "local_fits": fits,
            "condition_predictions": condition_predictions,
            "local_scope": "same frozen instrument, column serial, matrix, phase recipe and sampled gradient range only",
            "independent_validation": validation, "best_observed_unvalidated": best, "next_experiment": next_test,
            "observed_condition_quality": usable,
            "action_required": action_required,
            "separation_confirmed": False, "uncertainty_90_calibrated": False, "instrument_execution_authorized": False,
            "limitations": ["Local median residual correction is not a universal QSRR or width model",
                            "No claimed calibrated interval from training residuals or two validation conditions",
                            "No predicted Rs; grid scheduler precedes Bayesian optimization",
                            "Full method suitability criteria and prospective experiments remain required"]}


def write_guide(out, protocol, summary):
    text = """# Цикл TFV/FTC: прогноз → эксперимент → корректировка

Это инфраструктура для реальных измерений, не доказанное улучшение модели.
Фактическое число принятых измерений и состояние локальной модели приведены в конце документа.

## План для согласования с аналитиком

Одна колонка, неизменные поток и температура. Шесть условий адаптации:
два предложенных pH-уровня 2,6/3,6 × три длительности аналитического градиента (0,8/1,0/1,2).
Это черновые точки, не утверждённый рецепт. Изменяется только аналитическая часть;
время промывки, возврата и уравновешивания не масштабируется вместе с градиентом.
Ещё две промежуточные программы зарезервированы для независимой проверки;
их измерения никогда не входят в обучение и выбор локальной поправки.

В protocol.json аналитик задаёт точный прибор, серийный номер колонки, матрицу,
режим детектора, измеренный dwell volume, предел давления и критерии пригодности.
Нужно заполнить полные metadata фаз для обоих pH, общий recipe_family_id и способ
измерения pH. Рецептуры не генерируются добавлением произвольного pH к 0,1% кислоты.
Исходные 0385/0386 различаются также буфером и долей метанола в A — это НЕ чистый pH-эксперимент.
После проверки рецептур и уравновешивания аналитик подписывает протокол:
approval_status=approved, analyst=<имя>, equilibration_confirmed=true.

## Как работать

1. До опытов выполнить predict: сохраняются прогнозы CatBoost, хэш модели и точные
   снимки методов с прибором. Dwell volume пока фиксирует область применимости:
   базовая модель не обучена его физической поправке, и время нельзя просто сдвинуть на V/F.
   Повторный predict не перезаписывает зафиксированный прогноз. Новый прибор, протокол
   или модель требуют отдельной папки кампании; задним числом прогноз не фиксируется.
2. Выполнить холостой раствор, отдельные стандарты, смесь и повторы после согласования с аналитиком.
3. Для каждой инъекции заполнить observation_template.json и загрузить через команду
   record --observation путь/к/инъекции.json. После проверки строка добавится в observations.jsonl.
   actual_method скопировать из predictions.json только если реальные
   условия действительно совпали. Иначе нужен отдельный протокол и новый прогноз.
4. Указать реальный исходный сигнал, SHA256 и идентификатор инъекции; reviewer одобряет строку.
   Неудачные опыты сохраняются: coeluted/not_detected/unidentified без выдуманного tR/ширины.
5. Выполнить update. При трёх различных условиях на рецептуру/вещество можно оценить
   медианную поправку к tR. Повторные инъекции усредняются внутри условия, а не увеличивают
   фиктивно размер обучающей выборки. Поправка применяется только если leave-condition-out
   проверка внутри адаптации лучше исходного прогноза. Зарезервированная валидация не участвует в выборе.

Rs берётся из фактического отчёта либо рассчитывается из tR и ширин двух идентифицированных
пиков одной инъекции. Два tR без ширин не превращаются в Rs. Выбор лучшего наблюдённого
условия использует минимум Rs по повторам и проверку давления. Это ещё не валидация метода:
критерии удерживания, сигнала, асимметрии, переноса и воспроизводимости задаёт аналитик.
Система пока заполняет информативный план опытов и предлагает подтверждающие инъекции,
не выдаёт фиктивно обученный байесовский оптимизатор или низкую неопределённость.

## Текущее состояние

"""
    text += "```json\n"+json.dumps(summary, ensure_ascii=False, indent=2)+"\n```\n"
    (out/"CYCLE.md").write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["init", "predict", "record", "update"])
    p.add_argument("--observation", type=Path, help="Reviewed JSON for one real injection; required by record")
    p.add_argument("--folder", type=Path, default=Path("data/lab/tfv_ftc"))
    p.add_argument("--recommendation", type=Path, default=Path("reports/customer_multicolumn/lab_recommendation.json"))
    p.add_argument("--comparison", type=Path, default=Path("reports/customer_multicolumn/comparison.json"))
    p.add_argument("--out", type=Path, default=Path("reports/lab_cycle"))
    args=p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    protocol = initialize(json.loads(args.recommendation.read_text()), args.folder) if args.command == "init" else json.loads((args.folder/"protocol.json").read_text())
    if args.command == "predict":
        freeze_predictions(args.folder, protocol, json.loads(args.comparison.read_text()))
    if args.command == "record":
        if not args.observation:
            p.error("record requires --observation path/to/reviewed_injection.json")
        import_observation(args.folder, json.loads(args.observation.read_text()))
    predictions_path = args.folder/"predictions.json"
    frozen = json.loads(predictions_path.read_text()) if predictions_path.exists() else {}
    records = [json.loads(line) for line in (args.folder/"observations.jsonl").read_text().splitlines() if line.strip()]
    summary = summarize(protocol, frozen, records)
    (args.out/"state.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_guide(args.out, protocol, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
