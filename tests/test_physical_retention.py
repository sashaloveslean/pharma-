"""Synthetic mathematical tests, not experimental validation of TFV/FTC."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from physical_retention import calibrate
from audit_physical_retention import program_kind


def measurements():
    return [{"observation_id": str(i), "full_inchikey": "SGOIRFVFHAKUTI-ZCFIWIBFSA-N",
             "phi": phi, "t0_min": 1., "tr_min": 1+10**(2-3*phi),
             "mode": "isocratic", "reviewed": True, "label_origin": "experiment",
             "t0_origin": "measured", "t0_evidence": "test fixture only", "source": "test fixture only",
             "split": "validation" if i == 3 else "calibration",
             "context": {"column": "test", "modifier": "methanol", "aqueous_recipe": "fixed",
                         "pH": 3., "temperature_c": 30., "flow_ml_min": .2,
                         "instrument_id": "test", "pH_convention": "aqueous"}}
            for i,phi in enumerate([.2,.4,.6,.3])]


class PhysicalRetentionTests(unittest.TestCase):
    def test_recovers_known_lss_and_independent_composition(self):
        result = calibrate(measurements())
        self.assertAlmostEqual(result["S"], 3)
        self.assertAlmostEqual(result["log10_kw"], 2)
        self.assertAlmostEqual(result["validation_mae_min"], 0)
        self.assertIsNone(result["predicted_Rs"])
        self.assertFalse(result["gradient_supported"])

    def test_validation_does_not_change_fit(self):
        rows = measurements()
        rows[-1]["tr_min"] = 1000
        self.assertAlmostEqual(calibrate(rows)["S"], 3)

    def test_no_extrapolation(self):
        rows = measurements(); rows[-1]["phi"] = .9
        self.assertIsNone(calibrate(rows)["validation"][0]["predicted_tr_min"])

    def test_evidence_and_context_guards(self):
        for key,value in [("t0_origin","estimated"),("label_origin","catboost"),
                          ("mode","gradient"),("reviewed",False),("tr_min",.5),
                          ("phi",float("nan")),("observation_id","1")]:
            with self.subTest(key=key):
                rows = measurements(); rows[0][key] = value
                with self.assertRaises(ValueError): calibrate(rows)
        rows = measurements(); rows[0]["context"]["pH"] = 5
        with self.assertRaises(ValueError): calibrate(rows)

    def test_repeats_are_not_distinct_conditions_and_no_split_overlap(self):
        rows = measurements(); rows[1]["phi"] = rows[0]["phi"]
        with self.assertRaises(ValueError): calibrate(rows)
        rows = measurements(); rows[-1]["phi"] = rows[0]["phi"]
        with self.assertRaises(ValueError): calibrate(rows)

    def test_missing_program_is_not_isocratic(self):
        p = {"t [min]": 0, "A [%]": 90, "B [%]": 10, "C [%]": 0,
             "D [%]": 0, "flow rate [ml/min]": .2}
        self.assertEqual(program_kind([p]), "incomplete_program")
        q = {**p, "t [min]": 10}
        self.assertEqual(program_kind([p,q]), "constant_composition_candidate")
        q.update({"A [%]": 10, "B [%]": 90})
        self.assertEqual(program_kind([p,q]), "gradient")
        q["B [%]"] = float("nan")
        self.assertEqual(program_kind([p,q]), "invalid_program")


if __name__ == "__main__":
    unittest.main()
