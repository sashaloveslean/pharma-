from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recommend_lab import choose_experiments
from propose_pair_method import SPEC


class LabRecommendationTests(unittest.TestCase):
    def candidate(self, sid, col, verified=False, gap=0):
        return {"candidate_id": sid+":time_x1.0", "source_system": sid, "column": col,
                "manufacturer_limits_checked": verified, "origin": "source_program", "time_scale": 1,
                "conditions": {}, "gradient_program": [
                    {"t [min]": t, "A [%]": a, "B [%]": 100-a, "C [%]": 0, "D [%]": 0}
                    for t,a in [(0,90), (10,10), (11,90), (15,90)]], "time_gap_min": gap,
                "same_system_training_rows": 10, "metadata_source": "https://example.org/method",
                "method_source": [{"source": "Publication - author", "url": "https://example.org/paper"}]}

    def test_checked_source_beats_large_ml_gap_and_backups_use_other_columns(self):
        primary = self.candidate("report:0386", SPEC["name"], True)
        same_column = self.candidate("report:0385", SPEC["name"], True, 100)
        others = [self.candidate("a", "Column A", gap=1000), self.candidate("b", "Column B", gap=2)]
        comparison = {"candidates": [*others, same_column, primary], "rank_status": "UNRESOLVED"}
        result = choose_experiments(comparison, lambda _: [])
        self.assertEqual(result["primary"]["candidate"]["source_system"], "report:0386")
        self.assertEqual(len(result["backups"]), 2)
        self.assertEqual(len({r["candidate"]["column"] for r in [result["primary"], *result["backups"]]}), 3)
        self.assertFalse(result["instrument_execution_authorized"])
        self.assertFalse(result["primary"]["separation_confirmed"])

    def test_no_valid_source_means_blocked_not_a_fabricated_recipe(self):
        c = self.candidate("bad", "Column")
        result = choose_experiments({"candidates": [c]}, lambda _: ["missing_gradient"])
        self.assertIsNone(result["primary"])
        self.assertEqual(result["status"], "BLOCKED_NO_COMPLETE_SOURCE_METHOD")
        self.assertEqual(result["rejected"][0]["reasons"], ["missing_gradient"])

    def test_generated_variants_are_not_mislabeled_as_source_programs(self):
        c = self.candidate("scaled", "Column")
        c["time_scale"] = .8
        c["origin"] = "generated_time_scaled_program"
        result = choose_experiments({"candidates": [c]}, lambda _: [])
        self.assertIsNone(result["primary"])

    def test_missing_return_or_hold_cannot_become_primary(self):
        c = self.candidate("a", "Column A")
        c["gradient_program"] = c["gradient_program"][:2]
        result = choose_experiments({"candidates": [c]}, lambda _: [])
        self.assertIsNone(result["primary"])
        c["gradient_program"] = self.candidate("a", "Column A")["gradient_program"][:-1]
        result = choose_experiments({"candidates": [c]}, lambda _: [])
        self.assertIsNone(result["primary"])

    def test_selection_does_not_mutate_predictions_or_claim_backup_passports(self):
        c = self.candidate("a", "Column A")
        before = deepcopy(c)
        result = choose_experiments({"candidates": [c]}, lambda _: [])
        self.assertEqual(c, before)
        self.assertIsNone(result["primary"]["column_spec"])
        self.assertTrue(any("паспорт" in note for note in result["primary"]["preflight"]))


if __name__ == "__main__":
    unittest.main()
