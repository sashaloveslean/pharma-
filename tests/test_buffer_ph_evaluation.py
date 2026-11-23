import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_buffer_ph import numeric_keys

class PhFeatureTests(unittest.TestCase):
    def test_no_response_or_ph_leakage(self):
        rows=[{'metadata':{'id':'12','column.t0':1.2,'eluent.A.pH':3,'eluent.B.pH':4,'eluent.A.nh4form':5,'column.length':100}}]
        self.assertEqual(numeric_keys(rows),['column.length','eluent.A.nh4form'])

if __name__=='__main__':unittest.main()
