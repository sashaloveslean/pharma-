from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_mcmrt_transfer import encode_target,decode_target
from audit_early_exit import gate


class EarlyExitTests(unittest.TestCase):
    def test_roundtrip_positive_early_and_late_times(self):
        times=np.array([.01,.5,3.,50.]);scales=np.array([7.,7.,20.,70.])
        for mode in ['analytical_scaled','analytical_log']:
            np.testing.assert_allclose(decode_target(encode_target(times,scales,mode),scales,mode),times)

    def test_invalid_measured_times_never_clipped_into_labels(self):
        for value in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):encode_target([value],[10],'analytical_log')

    def test_improving_early_does_not_excuse_large_overall_degradation(self):
        previous={'early':{'n':20,'mae_min':2.},'all':{'mae_min':1.}}
        candidate={'early':{'n':20,'mae_min':1.},'all':{'mae_min':1.2}}
        self.assertFalse(gate(previous,candidate))
        candidate['all']['mae_min']=1.04
        self.assertTrue(gate(previous,candidate))
        candidate['early']['mae_min']=2.1
        self.assertFalse(gate(previous,candidate))
