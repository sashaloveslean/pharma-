import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_realism import split_groups, group_radius
from realistic_features import ChargeCache, enriched_vector, chemistry, PATTERNS


class ScientificValidationTests(unittest.TestCase):
    def test_four_part_split_has_no_molecule_overlap(self):
        groups = np.asarray([f"molecule{i}" for i in range(40) for _ in range(3)])
        splits = split_groups(groups)
        self.assertEqual(sum(map(len, splits)), len(groups))
        for i, left in enumerate(splits):
            for right in splits[i+1:]:
                self.assertFalse(set(groups[left]) & set(groups[right]))

    def test_calibration_uses_worst_residual_per_molecule(self):
        groups = np.asarray(["a", "a", "b", "c"])
        radius = group_radius(np.asarray([10, 0, 9, 8]), np.zeros(4), groups, alpha=0.25)
        self.assertEqual(radius, 10)
        self.assertTrue(math.isinf(group_radius(np.ones(4), np.zeros(4), groups)))

    def test_missing_ph_is_not_encoded_as_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = ChargeCache(Path(directory) / "cache.json")
            self.assertTrue(all(math.isnan(v) for v in cache.get("CCO", None)))
            self.assertFalse(cache.records)

    def test_premixed_eluent_is_translated_into_actual_solvent_fraction(self):
        row = {"smiles": "CCO", "metadata": {"column.length": 100, "column.id": 2.1,
               "column.flowrate": 0.2, "eluent.A.h2o": 90, "eluent.A.meoh": 10,
               "eluent.A.acn": 0, "eluent.B.h2o": 0, "eluent.B.meoh": 100, "eluent.B.acn": 0},
               "gradient_program": [{"t [min]": 0, "A [%]": 99, "B [%]": 1,
                                      "C [%]": 0, "D [%]": 0, "flow rate [ml/min]": 0.2}]}
        with tempfile.TemporaryDirectory() as directory:
            values = enriched_vector(row, [], ChargeCache(Path(directory) / "cache.json"))
        actual_solvents = values[-32:-2]
        self.assertAlmostEqual(actual_solvents[0], 89.1)
        self.assertAlmostEqual(actual_solvents[10], 10.9)

    def test_tenofovir_phosphonic_acid_is_explicitly_represented(self):
        _, groups = chemistry("C[C@H](Cn1cnc2c(N)ncnc21)OCP(=O)(O)O")
        self.assertEqual(groups[list(PATTERNS).index("PhosphonicAcid")], 1)


if __name__ == "__main__":
    unittest.main()
