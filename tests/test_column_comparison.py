from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_columns import method_issues, select_model


class ColumnComparisonTests(unittest.TestCase):
    def method(self):
        return {"metadata": {"column.name": "Example C18", "column.length": "100", "column.id": "2.1",
                             "column.particle.size": "2", "column.temperature": "30", "eluent.A.pH": "3",
                             "eluent.A.h2o": "100", "eluent.B.acn": "100"},
                "gradient_program": [{"t [min]": "0", "A [%]": "90", "B [%]": "10", "C [%]": "0",
                                      "D [%]": "0", "flow rate [ml/min]": "0.3"},
                                     {"t [min]": "10", "A [%]": "10", "B [%]": "90", "C [%]": "0",
                                      "D [%]": "0", "flow rate [ml/min]": "0.3"}]}

    def test_channel_and_solvent_mass_balances_checked_separately(self):
        original = self.method()
        self.assertEqual(method_issues(original), [])
        broken = deepcopy(original)
        broken["gradient_program"][0]["A [%]"] = "89.91"
        self.assertIn("gradient_mass_balance", method_issues(broken))
        broken = deepcopy(original)
        broken["metadata"]["eluent.A.h2o"] = "99"
        self.assertIn("active_channel_solvent_balance", method_issues(broken))

    def test_incomplete_reference_is_rejected_not_filled_with_defaults(self):
        method = self.method()
        method["metadata"]["column.temperature"] = ""
        self.assertIn("missing_or_invalid_column.temperature", method_issues(method))

    def test_source_method_label_cannot_silently_change_column_identity(self):
        method = self.method()
        method["metadata"]["column.name"] = "Restek Raptor Biphenyl"
        method["source_info"] = [{"name": "MSMS_pos_40_MeOH_CORTECS C18"}]
        self.assertIn("source_method_label_disagrees_with_column_metadata", method_issues(method))

    def test_model_choice_uses_selection_not_target_or_test_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"evaluation.json"
            report = {"dataset_sha256": "digest", "models": {
                "a": {"selection": {"macro_group_mae_min": 1}, "test": {"mae_min": 100}},
                "b": {"selection": {"macro_group_mae_min": 2}, "test": {"mae_min": 0}}},
                "split_observation_ids": {"selection": ["v"], "calibration": ["c"], "test": ["t"]}}
            path.write_text(json.dumps(report))
            selected, _ = select_model([path])
            self.assertEqual(selected[2], "a")
            second = Path(directory)/"changed.json"
            report["split_observation_ids"]["test"] = ["different-test"]
            second.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                select_model([path, second])


if __name__ == "__main__":
    unittest.main()
