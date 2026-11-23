from copy import deepcopy
import math
import unittest
from test_mcmrt import record
from early_retention_features import physical_features
from audit_physical_features import gate


class PhysicalFeatureTests(unittest.TestCase):
    def test_geometry_flow_and_actual_solvent(self):
        row = record(); a = physical_features(row)
        volume = math.pi * 2.1**2 / 4 * 100 / 1000
        self.assertAlmostEqual(a['geometric_volume_over_flow_min'], volume/.3)
        self.assertAlmostEqual(a['initial_meoh_fraction'], .1)
        for p in row['gradient_program']: p['flow rate [ml/min]'] *= 2
        b = physical_features(row)
        self.assertAlmostEqual(b['geometric_volume_over_flow_min'], a['geometric_volume_over_flow_min']/2)
        self.assertAlmostEqual(b['superficial_velocity_mm_s'], a['superficial_velocity_mm_s']*2)

    def test_labels_and_late_wash_do_not_change_added_features(self):
        row = record(); a = physical_features(row)
        row.update(retention_time_min=99, t0=77, classification='early')
        wash = deepcopy(row['gradient_program'][-1]); wash['t [min]'] = 30
        row['gradient_program'].append(wash)
        self.assertEqual(a, physical_features(row))

    def test_invalid_flow_rejected(self):
        row = record(); row['gradient_program'][0]['flow rate [ml/min]'] = 0
        with self.assertRaises(ValueError): physical_features(row)

    def test_gate_requires_every_scenario(self):
        joint = 'unseen_column_and_molecules'
        part = {'all': {'mae_min': 1}, 'early': {'n': 5, 'mae_min': 2}}
        previous = {joint: deepcopy(part), 'unseen_method': deepcopy(part)}
        candidate = deepcopy(previous); candidate[joint]['early']['mae_min'] = 1
        config = {'max_overall_mae_ratio_each_scenario': 1.05}
        self.assertTrue(gate(previous, candidate, config))
        candidate['unseen_method']['all']['mae_min'] = 1.06
        self.assertFalse(gate(previous, candidate, config))
