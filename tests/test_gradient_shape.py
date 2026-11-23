import sys
from pathlib import Path
from copy import deepcopy
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_gradient_shape import shape_features, adapt
from test_mcmrt import record

class ShapeTests(unittest.TestCase):
    def test_constant_residual(self):
        self.assertAlmostEqual(adapt([[0,1],[1,2]],[3,3],[9,10]),3)
    def test_shape_changes_at_fixed_duration(self):
        a=record();b=deepcopy(a)
        b['gradient_program'].insert(1,{'t [min]':5.,'A [%]':90.,'B [%]':10.,'flow rate [ml/min]':.3})
        self.assertFalse(np.array_equal(shape_features(a),shape_features(b)))
    def test_wash_and_response_not_features(self):
        a=record();expected=shape_features(a);a['retention_time_min']=999
        p=deepcopy(a['gradient_program'][-1]);p['t [min]']=50;a['gradient_program'].append(p)
        np.testing.assert_equal(expected,shape_features(a))
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):adapt([[0],[1]],[2,float('nan')],[2])
