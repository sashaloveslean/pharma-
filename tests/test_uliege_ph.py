import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from curate_uliege_ph import annotate

class UliegePhTests(unittest.TestCase):
    def test_target_label_not_used_as_measured(self):
        r=annotate({'pH_target':2,'pH_measured':2.67})
        self.assertEqual(r['phase_recipe']['nominal_ph'],2.7)
        self.assertEqual(r['measured_pH'],2.67)
        self.assertFalse(r['is_training_ready'])
    def test_unknown_recipe_not_filled(self):
        r=annotate({'pH_target':4,'pH_measured':4.1})
        self.assertIsNone(r['phase_recipe'])
        self.assertIsNone(r['reported_buffer_concentration_mM'])
    def test_measured_missing_not_inferred(self):
        self.assertIsNone(annotate({'pH_target':5,'pH_measured':None})['measured_pH'])

if __name__=='__main__':unittest.main()
