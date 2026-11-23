import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_ph_delta import unique_outcome

class PairTests(unittest.TestCase):
    def test_distinct_outcomes_not_silently_averaged(self):
        self.assertIsNone(unique_outcome([{'retention_time_min':1,'smiles':'C'},{'retention_time_min':2,'smiles':'C'}]))
    def test_duplicate_outcome_is_single(self):
        row={'retention_time_min':1,'smiles':'C'}
        self.assertEqual(unique_outcome([row,row]),row)
    def test_structure_conflict_rejected(self):
        self.assertIsNone(unique_outcome([{'retention_time_min':1,'smiles':'C'},{'retention_time_min':1,'smiles':'CC'}]))

if __name__=='__main__':unittest.main()
