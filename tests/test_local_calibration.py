import sys
from pathlib import Path
import unittest
from copy import deepcopy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_local_calibration import partition, fit_offsets, connectivity, TARGETS


class LocalCalibrationTests(unittest.TestCase):
    def rows(self):
        return [dict(system_id=s, compound_id=m+'-stereo', observation_id=s+m,
                     observed_tr_min=float(i+2), scaled_tr_min=float(i))
                for s in ('a', 'b') for i, m in enumerate(
                    [f'm{i}' for i in range(50)]+sorted(TARGETS))]

    def test_molecular_disjointness_and_targets(self):
        selected, fit, test = partition(self.rows(), {'a': 'same', 'b': 'same'})
        self.assertEqual(len(selected['same']), 20)
        self.assertFalse({connectivity(r) for r in fit} & {connectivity(r) for r in test})
        self.assertFalse({connectivity(r) for r in fit} & TARGETS)
        self.assertTrue(TARGETS <= {connectivity(r) for r in test})

    def test_selection_independent_of_labels_and_order(self):
        rows = self.rows(); ids = {'a': 'one', 'b': 'two'}
        before = partition(rows, ids)[0]
        for r in rows: r['observed_tr_min'] = 999; r['scaled_tr_min'] = -10
        self.assertEqual(before, partition(list(reversed(rows)), ids)[0])

    def test_offset_and_equal_molecule_weight(self):
        rows = self.rows()[:3]
        rows[0]['observed_tr_min'] = 1000
        expanded = rows + [deepcopy(rows[0]) for _ in range(100)]
        self.assertEqual(fit_offsets(expanded, {'a': 'method'})['method'], 2.)
