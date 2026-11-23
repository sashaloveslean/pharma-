"""Tests of fitting/splits; toy measurements do not become source observations."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_lss_training import fit_series


class LSSTrainingTests(unittest.TestCase):
    def rows(self):
        return [{"phi":p,"k":10**(2-3*p)} for p in [.1,.2,.3,.4,.5,.6,.7,.8]]

    def test_parameters_and_composition_split(self):
        r=fit_series(self.rows())
        self.assertAlmostEqual(r["log10_kw"],2)
        self.assertAlmostEqual(r["S"],3)
        self.assertAlmostEqual(r["validation_mae_log10_k"],0)
        self.assertFalse(set(r["calibration_phi"]) & set(r["validation_phi"]))
        self.assertTrue(r["kw_is_extrapolated_to_water"])

    def test_validation_labels_never_enter_fit(self):
        rows=self.rows();base=fit_series(rows)
        for r in rows:
            if r["phi"] in base["validation_phi"]:r["k"]*=20
        actual=fit_series(rows)
        self.assertAlmostEqual(actual["S"],base["S"])
        self.assertAlmostEqual(actual["validation_mae_log10_k"],math.log10(20))

    def test_replicates_do_not_change_weight_or_distinct_compositions(self):
        rows=self.rows();rows += [dict(rows[0]) for _ in range(50)]
        self.assertAlmostEqual(fit_series(rows)["S"],3)
        self.assertIsNone(fit_series([{"phi":.2,"k":10}]*50))

    def test_invalid_data_rejected(self):
        for phi,k in [(float("nan"),1),(.1,0),(.1,-1),(1.1,1),(.1,float("inf"))]:
            with self.assertRaises(ValueError):fit_series([{"phi":phi,"k":k}])


if __name__=="__main__":unittest.main()
