import sys
from pathlib import Path
from copy import deepcopy
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_gradient_changes import method_pairs, summarize


class GradientChangeTests(unittest.TestCase):
    def test_pairs_exclude_condition_confounders(self):
        a = {'eligible_for_rt_research': True, 'metadata': {'column.name': 'C18', 'temperature': 30},
             'instrument': 'instrument', 'gradient_program': [{'t [min]': 0, 'flow rate [ml/min]': .2},
                                                            {'t [min]': 10, 'flow rate [ml/min]': .2}]}
        b = deepcopy(a); b['gradient_program'][-1]['t [min]'] = 20
        self.assertEqual(method_pairs({'a': a, 'b': b}), [('a', 'b')])
        for field, value in [('temperature', 40), ('column.name', 'C8')]:
            c = deepcopy(b); c['metadata'][field] = value
            self.assertEqual(method_pairs({'a': a, 'b': c}), [])
        c = deepcopy(b); c['gradient_program'][-1]['flow rate [ml/min]'] = .3
        self.assertEqual(method_pairs({'a': a, 'b': c}), [])
        c = deepcopy(b); c['instrument'] = 'other'
        self.assertEqual(method_pairs({'a': a, 'b': c}), [])

    def test_change_metrics_and_zero_prediction(self):
        rows = [dict(observed_change_min=2., predicted_change_min=0., time_scaling_change_min=1.)]
        m = summarize(rows)
        self.assertEqual(m['delta_mae_min'], 2.)
        self.assertEqual(m['direction_accuracy'], 0.)
        self.assertEqual(m['time_scaling_direction_accuracy'], 1.)
        self.assertEqual(summarize([]), {'n': 0})
