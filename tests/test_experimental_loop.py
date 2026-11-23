"""Synthetic tests only: no generated values are laboratory observations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from experimental_loop import (initialize, scaled_ramp, protocol_issues, actual_method, digest,
                               summarize, validate_observations, measured_rs, FULL_KEYS,
                               freeze_predictions, import_observation)
from propose_pair_method import SPEC


class ExperimentalLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        metadata = {"column.name": SPEC["name"], "column.length": "100", "column.id": "2.1",
                    "column.particle.size": "2.2", "column.temperature": "30", "column.flowrate": "0.2",
                    "column.t0": "1.1025", "eluent.A.h2o": "100", "eluent.A.pH": "2.6",
                    "eluent.B.meoh": "100", "eluent.B.pH": "2.6"}
        program = [{"t [min]": t, "A [%]": 100-b, "B [%]": b, "C [%]": 0, "D [%]": 0,
                    "flow rate [ml/min]": .2} for t,b in [(0,10),(4,50),(17,95),(25,95),(25.1,10),(30,10)]]
        candidate = {"conditions": metadata, "gradient_program": program, "method_source": [{"name": "SYNTHETIC TEST ONLY"}]}
        self.protocol = initialize({"primary": {"candidate": candidate}}, self.folder)
        self.protocol.update(approval_status="approved", analyst="SYNTHETIC TEST ONLY", equilibration_confirmed=True)
        self.protocol["instrument"].update(instrument_id="SYNTHETIC", column_serial="synthetic", matrix_id="test",
            detection_protocol_id="test", dwell_volume_ml=.1, dwell_measurement_method="synthetic", pressure_limit_bar=400)
        self.protocol["sample"].update(injection_ul=1, tfv_concentration_ug_ml=10, ftc_concentration_ug_ml=10, diluent_recipe="synthetic")
        for recipe in self.protocol["recipes"]:
            recipe.update(metadata={k:v for k,v in metadata.items() if k.startswith("eluent.")},
                          recipe_family_id="synthetic", ph_measurement_convention="synthetic aqueous")
            recipe["metadata"]["eluent.A.pH"] = recipe["proposed_pH"]
            recipe["metadata"]["eluent.B.pH"] = recipe["proposed_pH"]
        self.frozen = {"status": "FROZEN_BASE_PREDICTIONS", "protocol_sha256": digest(self.protocol), "predictions": {}}
        for e in self.protocol["experiments"]:
            method = actual_method(self.protocol, e)
            self.frozen["predictions"][e["experiment_id"]] = {"method_sha256": digest(method),
                "tr_min": {"tenofovir": 4*e["ramp_scale"], "emtricitabine": 5*e["ramp_scale"]}}

    def observation(self, index, suffix="", failed=False):
        e = self.protocol["experiments"][index]
        run_id = f"synthetic-{index}-{suffix}"
        path = self.folder/(run_id+".txt")
        path.write_text("SYNTHETIC TEST ONLY "+run_id)
        peaks = {}
        for name, key in FULL_KEYS.items():
            base = self.frozen["predictions"][e["experiment_id"]]["tr_min"][name]
            peaks[name] = {"inchikey": key, "status": "coeluted" if failed else "identified",
                "tr_min": None if failed else base+1, "width_half_height_min": None if failed else .2,
                "tailing": 1.1, "snr": 100}
        return {"run_id": run_id, "experiment_id": e["experiment_id"], "sample_type": "mixture", "review_status": "approved",
                "actual_method": actual_method(self.protocol, e), "raw_signal_path": str(path),
                "raw_signal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "raw_signal_record_id": run_id,
                "peaks": peaks, "pressure_bar": 100, "t0_min": 1, "t0_measurement_method": "synthetic",
                "reported_rs": None, "carryover_percent": .01}

    def test_simulation_cannot_enter_real_lab_log(self):
        row = self.observation(0)
        row["label_origin"] = "simulation"
        accepted, rejected = validate_observations(self.protocol, self.frozen, [row])
        self.assertEqual(accepted, [])
        self.assertIn("synthetic_or_predicted_label_is_not_a_lab_measurement", rejected[0]["issues"])

    def test_plan_is_draft_empty_and_never_overwrites_owned_protocol(self):
        original = json.loads((self.folder/"protocol.json").read_text())
        self.assertEqual(original["approval_status"], "draft")
        self.assertEqual(len([e for e in original["experiments"] if e["role"] == "adaptation"]), 6)
        self.assertEqual(len([e for e in original["experiments"] if e["role"] == "validation"]), 2)
        self.assertEqual((self.folder/"observations.jsonl").read_text(), "")
        with self.assertRaises(ValueError): initialize({"primary": {"candidate": {}}}, self.folder)

    def test_frozen_predictions_idempotent_and_never_rewritten(self):
        self.frozen["model_sha256"] = "test-model"
        comparison = {"model_sha256": "test-model"}
        freeze_predictions(self.folder, self.protocol, comparison, predictor=lambda *_: self.frozen)
        before = (self.folder/"predictions.json").read_bytes()
        freeze_predictions(self.folder, self.protocol, comparison, predictor=lambda *_: self.fail("Must not recompute"))
        self.assertEqual(before, (self.folder/"predictions.json").read_bytes())
        changed = deepcopy(self.protocol); changed["analyst"] = "changed"
        with self.assertRaises(ValueError): freeze_predictions(self.folder, changed, comparison)
        self.assertEqual(before, (self.folder/"predictions.json").read_bytes())

    def test_cannot_freeze_retrospective_predictions(self):
        (self.folder/"observations.jsonl").write_text('{"recorded": true}\n')
        with self.assertRaises(ValueError):
            freeze_predictions(self.folder, self.protocol, {"model_sha256": "test"}, predictor=lambda *_: self.frozen)

    def test_record_import_is_append_only_and_rejects_duplicate(self):
        (self.folder/"protocol.json").write_text(json.dumps(self.protocol))
        (self.folder/"predictions.json").write_text(json.dumps(self.frozen))
        row = self.observation(0)
        import_observation(self.folder, row)
        before = (self.folder/"observations.jsonl").read_bytes()
        with self.assertRaises(ValueError): import_observation(self.folder, row)
        self.assertEqual(before, (self.folder/"observations.jsonl").read_bytes())

    def test_only_ramp_changes_not_wash_or_equilibration(self):
        program = self.protocol["source_candidate"]["gradient_program"]
        scaled = scaled_ramp(program, .8)
        self.assertAlmostEqual(scaled[2]["t [min]"], 13.6)
        for i in range(3,len(program)):
            self.assertAlmostEqual(scaled[i]["t [min]"]-scaled[i-1]["t [min]"],
                                   program[i]["t [min]"]-program[i-1]["t [min]"])

    def test_protocol_demands_real_context_and_does_not_import_source_t0(self):
        self.assertEqual(protocol_issues(self.protocol), [])
        self.assertNotIn("column.t0", actual_method(self.protocol, self.protocol["experiments"][0])["metadata"])
        self.protocol["instrument"]["dwell_volume_ml"] = None
        self.assertIn("missing_measured_dwell_volume", protocol_issues(self.protocol))

    def test_ph_only_factor_cannot_silently_change_solvent_recipe(self):
        self.protocol["recipes"][1]["metadata"].update({"eluent.A.h2o": 90, "eluent.A.meoh": 10})
        self.assertIn("solvent_composition_changes_between_ph_levels", protocol_issues(self.protocol))

    def test_no_experiments_means_no_fitted_model(self):
        state = summarize(self.protocol, self.frozen, [])
        self.assertFalse(state["local_model_fitted"])
        self.assertEqual(state["status"], "WAITING_FOR_MEASUREMENTS")
        self.assertEqual(state["next_experiment"]["experiment_id"], "pH_low_ramp_1.0")
        self.assertFalse(state["instrument_execution_authorized"])

    def test_validation_outlier_never_changes_training_bias(self):
        rows = [self.observation(i) for i in range(6)]
        validation = self.observation(6)
        for p in validation["peaks"].values(): p["tr_min"] += 100
        state = summarize(self.protocol, self.frozen, rows+[validation])
        self.assertTrue(state["local_model_fitted"])
        self.assertAlmostEqual(state["local_fits"]["pH_low:tenofovir"]["bias_min"], 1)
        self.assertEqual(len(state["independent_validation"]), 2)
        self.assertGreater(state["independent_validation"][0]["corrected_absolute_error_min"], 90)
        self.assertIsNone(state["condition_predictions"]["pH_low_ramp_1.0"]["tenofovir"]["interval_90_min"])

    def test_replicates_do_not_fake_three_independent_conditions(self):
        rows = [self.observation(0, str(i)) for i in range(10)]
        state = summarize(self.protocol, self.frozen, rows)
        self.assertFalse(state["local_model_fitted"])

    def test_changed_dwell_or_wrong_identity_rejected_not_pooled(self):
        row = self.observation(0)
        row["actual_method"]["instrument"]["dwell_volume_ml"] = .2
        accepted, rejected = validate_observations(self.protocol, self.frozen, [row])
        self.assertFalse(accepted)
        self.assertIn("actual_method_or_instrument_differs_from_frozen_prediction", rejected[0]["issues"])
        self.assertEqual(self.protocol["instrument"]["dwell_volume_ml"], .1)

    def test_failed_experiments_are_preserved_without_fake_rt_or_rs(self):
        row = self.observation(0, failed=True)
        state = summarize(self.protocol, self.frozen, [row])
        self.assertEqual(state["accepted_runs"], 1)
        self.assertIn(row["run_id"], state["failed_runs_preserved"])
        self.assertIsNone(measured_rs(row))
        self.assertFalse(state["local_model_fitted"])
        row["peaks"]["tenofovir"]["tr_min"] = 0
        self.assertFalse(validate_observations(self.protocol, self.frozen, [row])[0])

    def test_two_times_without_widths_never_become_rs(self):
        row = self.observation(0)
        for p in row["peaks"].values(): p["width_half_height_min"] = None
        self.assertIsNone(measured_rs(row))

    def test_duplicate_raw_injection_and_wrong_stereochemistry_rejected(self):
        row = self.observation(0)
        copied = deepcopy(row); copied["run_id"] = "another-run-name"
        accepted, rejected = validate_observations(self.protocol, self.frozen, [row,copied])
        self.assertEqual(len(accepted), 1)
        self.assertIn("missing_or_duplicate_raw_injection", rejected[0]["issues"])
        row["peaks"]["emtricitabine"]["inchikey"] = "XQSPYNMVSIKCOC-PRJDIBJQSA-N"
        self.assertFalse(validate_observations(self.protocol, self.frozen, [row])[0])

    def test_blank_is_not_a_completed_screening_condition(self):
        row = self.observation(1, failed=True); row["sample_type"] = "blank"
        for p in row["peaks"].values(): p["status"] = "not_detected"
        state = summarize(self.protocol, self.frozen, [row])
        self.assertEqual(state["failed_runs_preserved"], [])
        self.assertIsNone(state["best_observed_unvalidated"])

    def test_pressure_violation_is_never_best_observed(self):
        row = self.observation(0); row["pressure_bar"] = 500
        state = summarize(self.protocol, self.frozen, [row])
        self.assertIsNone(state["best_observed_unvalidated"])
        self.assertIsNone(state["next_experiment"])
        self.assertEqual(state["action_required"], "REVIEW_PRESSURE_LIMIT_VIOLATION_BEFORE_FURTHER_RUNS")

    def test_front_elution_blocks_separation_optimization(self):
        row = self.observation(0)
        row["peaks"]["tenofovir"]["tr_min"] = row["t0_min"]
        state = summarize(self.protocol, self.frozen, [row])
        self.assertEqual(state["action_required"], "REVIEW_RETENTION_BEFORE_OPTIMIZING_SEPARATION")
        self.assertIsNone(state["next_experiment"])

    def test_buffer_change_cannot_be_called_ph_only(self):
        self.protocol["recipes"][1]["metadata"].update({"eluent.A.nh4form": 5, "eluent.A.nh4form.unit": "mM"})
        self.assertIn("buffer_species_or_concentration_changes_between_ph_levels", protocol_issues(self.protocol))

    def test_modified_protocol_invalidates_frozen_predictions(self):
        self.protocol["sample"]["injection_ul"] = 2
        state = summarize(self.protocol, self.frozen, [])
        self.assertEqual(state["status"], "FROZEN_PREDICTIONS_REQUIRED")

    def test_changed_raw_signal_is_rejected(self):
        row = self.observation(0)
        Path(row["raw_signal_path"]).write_text("SYNTHETIC MODIFIED SIGNAL")
        accepted, rejected = validate_observations(self.protocol, self.frozen, [row])
        self.assertFalse(accepted)
        self.assertIn("raw_signal_missing_or_checksum_mismatch", rejected[0]["issues"])

    def test_conditions_cannot_leak_into_validation_with_another_identifier(self):
        extra = deepcopy(self.protocol["experiments"][0]); extra.update(experiment_id="alias", role="validation")
        self.protocol["experiments"].append(extra)
        self.assertIn("same_method_in_adaptation_and_validation", protocol_issues(self.protocol))


if __name__ == "__main__":
    unittest.main()
