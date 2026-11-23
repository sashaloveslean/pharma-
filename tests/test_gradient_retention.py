"""Synthetic numerical fixtures only, never exported as TFV/FTC evidence."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from gradient_retention import retention, fit_campaign, simulate, validate_system
from open_retention import load_observations


def system():
    s = {"reviewed": True, "time_reference": "sample_entry_at_column_inlet",
         "t0_column_min": 1., "dwell_time_min": .3, "extra_column_time_min": .1,
         "context": {"column": "test column", "modifier": "methanol", "phase_recipes": "test only",
                     "pH_convention": "aqueous", "pH": 3., "temperature_c": 30.,
                     "flow_ml_min": .2, "instrument_id": "test"}}
    for k in ("t0_column_min","dwell_time_min","extra_column_time_min"):
        s[k+"_evidence"] = "SYNTHETIC UNIT TEST ONLY"
    return s


def program(ramp=10):
    return [{"time_min": 0., "phi": .1}, {"time_min": ramp, "phi": .9}, {"time_min": 50., "phi": .9}]


def campaign():
    s = system()
    return {"system": s, "full_inchikey": "SGOIRFVFHAKUTI-ZCFIWIBFSA-N", "observations": [
        {"observation_id": str(i), "program": program(ramp), "tr_min": retention(2,3,program(ramp),s),
         "split": "validation" if i==3 else "calibration", "label_origin": "experiment",
         "reviewed": True, "source": "SYNTHETIC UNIT TEST ONLY"}
        for i,ramp in enumerate([5,10,20,15])]}


class GradientRetentionTests(unittest.TestCase):
    def test_isocratic_limit(self):
        p = [{"time_min": 0, "phi": .5},{"time_min": 100, "phi": .5}]
        s = system()
        self.assertAlmostEqual(retention(2,3,p,s), 1+10**.5+.1, places=8)
        s["dwell_time_min"] = 40
        self.assertAlmostEqual(retention(2,3,p,s), 1+10**.5+.1, places=8)

    def test_linear_gradient_matches_independent_closed_form(self):
        s = system(); s["dwell_time_min"] = 0; s["extra_column_time_min"] = 0
        p = [{"time_min": 0, "phi": .1},{"time_min": 20, "phi": .9}]
        a = math.log(10)*3*.04
        expected = 1+math.log1p(a*10**(2-3*.1))/a
        self.assertAlmostEqual(retention(2,3,p,s),expected,places=8)

    def test_delay_and_pregradient_elution(self):
        s = system(); p = program()
        original = retention(2,3,p,s)
        s["dwell_time_min"] = 5
        self.assertGreater(retention(2,3,p,s), original)
        s["dwell_time_min"] = 45
        self.assertAlmostEqual(retention(0,3,p,s),1+10**(-.3)+.1)

    def test_nonelution_not_fabricated(self):
        self.assertIsNone(retention(10,1,program(),system()))

    def test_recovers_parameters_and_separates_simulation(self):
        c = campaign(); model = fit_campaign(c)
        self.assertAlmostEqual(model["log10_kw"],2,places=5)
        self.assertAlmostEqual(model["S"],3,places=5)
        self.assertLess(model["validation_mae_min"],1e-5)
        result = simulate(model,program(12))
        self.assertEqual(result["label_origin"],"simulation")
        self.assertFalse(result["training_ready"])
        self.assertIsNone(result["predicted_Rs"])
        self.assertEqual(result["domain"],"new_program_unvalidated")

    def test_validation_does_not_fit_parameters(self):
        c = campaign(); c["observations"][-1]["tr_min"] = 30
        model = fit_campaign(c)
        self.assertAlmostEqual(model["S"],3,places=5)
        self.assertGreater(model["validation_mae_min"],10)

    def test_missing_measurements_and_synthetic_labels_blocked(self):
        for key in ("t0_column_min","dwell_time_min","extra_column_time_min"):
            s = system(); s[key] = None
            with self.assertRaises(ValueError): validate_system(s)
        c = campaign(); c["observations"][0]["label_origin"] = "simulation"
        with self.assertRaises(ValueError): fit_campaign(c)

    def test_holdout_overlap_and_repeats_blocked(self):
        c = campaign(); c["observations"][-1]["program"] = c["observations"][0]["program"]
        with self.assertRaises(ValueError): fit_campaign(c)
        c = campaign(); c["observations"][1]["program"] = c["observations"][0]["program"]
        with self.assertRaises(ValueError): fit_campaign(c)

    def test_unidentifiable_parameters_blocked(self):
        c = campaign()
        for i,r in enumerate(c["observations"]):
            r["program"] = [{"time_min": 0,"phi": .1},{"time_min": 40+i,"phi": .1}]
            r["tr_min"] = 20
        with self.assertRaises(ValueError): fit_campaign(c)

    def test_edited_model_rejected(self):
        m = fit_campaign(campaign()); m["S"] += 1
        with self.assertRaises(ValueError): simulate(m,program())

    def test_public_loader_rejects_synthetic_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"observations.jsonl"
            path.write_text(json.dumps({"system_id":"test","label_origin":"simulation"})+"\n")
            with self.assertRaises(ValueError): load_observations(path)


if __name__ == "__main__":
    unittest.main()
