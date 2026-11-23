import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_local_gradient_model import correction

class LocalGradientTests(unittest.TestCase):
    def test_constant_residual_is_preserved(self):
        value,slope=correction([5,10,20],[2,2,2],15)
        self.assertAlmostEqual(value,2);self.assertAlmostEqual(slope,0)
    def test_local_trend_and_regularization(self):
        scales=[5,10,20];residuals=3*np.log(scales)
        value,slope=correction(scales,residuals,10)
        self.assertAlmostEqual(value,3*np.log(10))
        self.assertGreater(slope,0);self.assertLess(slope,3)
    def test_two_distinct_program_scales_required(self):
        for scales,residuals in [([5],[2]),([5,5],[2,3])]:
            with self.assertRaises(ValueError):correction(scales,residuals,10)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):correction([5,10],[float('nan'),2],10)
