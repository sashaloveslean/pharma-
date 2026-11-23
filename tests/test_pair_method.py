import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from propose_pair_method import check_method, SPEC, possible_gap


class ColumnConstraintTests(unittest.TestCase):
    def test_overlapping_retention_intervals_do_not_imply_positive_separation(self):
        self.assertEqual(possible_gap([0, 7], [0, 8]), [0, 8])
        self.assertEqual(possible_gap([1, 2], [4, 6]), [2, 5])

    def setUp(self):
        self.meta = {"column.name": SPEC["name"], "column.length": 100, "column.id": 2.1,
                     "column.particle.size": 2.2, "column.temperature": 30,
                     "eluent.A.pH": 2.6, "eluent.A.h2o": 100}
        self.gradient = [{"t [min]": 0, "A [%]": 90, "B [%]": 10,
                          "C [%]": 0, "D [%]": 0, "flow rate [ml/min]": 0.2}]

    def test_valid_reference_passes_without_inventing_100_percent_water_compatibility(self):
        self.assertEqual(check_method(self.meta, self.gradient), [])
        self.gradient[0].update({"A [%]": 99, "B [%]": 1})
        self.assertIn("water_above_column_compatibility", check_method(self.meta, self.gradient))
        self.meta["eluent.A.h2o"] = 90
        self.assertEqual(check_method(self.meta, self.gradient), [])

    def test_unbalanced_percentages_are_not_silently_normalized(self):
        self.gradient[0].update({"A [%]": 0.01, "B [%]": 99.9})
        self.assertIn("gradient_mass_balance", check_method(self.meta, self.gradient))
        self.assertEqual(self.gradient[0]["A [%]"], 0.01)

    def test_geometry_and_flow_are_checked_against_selected_column(self):
        self.meta["column.id"] = 4.6
        self.gradient[0]["flow rate [ml/min]"] = 2
        errors = check_method(self.meta, self.gradient)
        self.assertIn("wrong_column_geometry", errors)
        self.assertIn("flow_outside_column_range", errors)


if __name__ == "__main__":
    unittest.main()
