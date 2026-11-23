import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from method_physics import linear_velocity_mm_s,bed_volume_ml,gradient_segments
from audit_method_efficiency import flow_transfer


class MethodPhysicsTests(unittest.TestCase):
    def test_units_and_unknown_porosity(self):
        self.assertAlmostEqual(linear_velocity_mm_s(0.2,2.1),0.9623881,places=6)
        self.assertAlmostEqual(linear_velocity_mm_s(0.2,2.1,0.68),0.9623881/0.68,places=6)
        with self.assertRaises(ValueError):
            linear_velocity_mm_s(0.2,2.1,0)

    def test_same_linear_velocity_requires_area_scaling(self):
        left=linear_velocity_mm_s(1,4.2)
        right=linear_velocity_mm_s(0.25,2.1)
        self.assertAlmostEqual(left,right)

    def test_gradient_roles_and_flow_transfer_preserve_ramp_and_wash_volumes(self):
        program=[{"t [min]":t,"A [%]":100-b,"B [%]":b,"C [%]":0,"D [%]":0,
                  "flow rate [ml/min]":0.2} for t,b in [(0,10),(4,50),(17,95),(25,95),(25.1,10),(30,10)]]
        before=gradient_segments(program)
        after=gradient_segments(flow_transfer(program,0.4,100,2.1))
        self.assertEqual([s["role"] for s in before], [s["role"] for s in after])
        for old,new in zip(before[:-1],after[:-1]):
            self.assertAlmostEqual(old["volume_ml"],new["volume_ml"])
        self.assertAlmostEqual(after[-1]["volume_ml"]/bed_volume_ml(100,2.1),5)
        self.assertAlmostEqual(before[0]["b_slope_percentage_points_min"],10)
        self.assertAlmostEqual(before[2]["duration_min"],8)


if __name__=="__main__":
    unittest.main()
