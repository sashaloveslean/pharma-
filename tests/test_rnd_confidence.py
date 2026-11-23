import sys
from pathlib import Path
import unittest
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rnd_confidence import perturb, confidence_level, support

class RndConfidenceTests(unittest.TestCase):
    def test_sensitivity_is_not_evidence_of_accuracy(self):
        self.assertEqual(confidence_level(True,False,True),'low')
        self.assertEqual(confidence_level(False,True,True),'low')
    def test_perturbation_semantics_and_no_mutation(self):
        p={'middle_B':50,'knee_min':4,'hold_min':1,'end_min':12}
        original=deepcopy(p)
        self.assertEqual(perturb(p,'middle_B',2)['middle_B'],52)
        self.assertAlmostEqual(perturb(p,'end_min',.05)['end_min'],12.6)
        self.assertAlmostEqual(perturb(p,'hold_min',-.1)['hold_min'],.9)
        self.assertEqual(p,original)
    def test_support_excludes_quarantined_methods(self):
        m={'column.name':'Column'}
        m.update({f'eluent.{ch}.{k}':0 for ch in 'AB' for k in ('h2o','meoh','acn','formic','nh4form','nh4ac')})
        source={'metadata':m}
        a={'metadata':deepcopy(m),'eligible_for_rt_research':False}
        b={'metadata':deepcopy(m),'eligible_for_rt_research':True};b['metadata']['column.name']='Other'
        r=support(source,{'a':a,'b':b})
        self.assertEqual(r['same_column_label_methods'],0)
        self.assertEqual(r['same_phase_recipe_methods'],1)
        self.assertFalse(r['local_empirical_validation'])
