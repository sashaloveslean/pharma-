import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_buffer_ph import phase_record, active_channels, signature_without_ph

class BufferAuditTests(unittest.TestCase):
    def test_unknown_unit_not_assumed_mM(self):
        p=phase_record({'eluent.A.nh4form':5}, 'A', 'report')
        self.assertIsNone(p['salts'][0]['concentration_mM'])
    def test_molar_conversion(self):
        p=phase_record({'eluent.A.nh4ac':.01,'eluent.A.nh4ac.unit':'M'},'A','report')
        self.assertEqual(p['salts'][0]['concentration_mM'],10)
    def test_no_ph_inference_from_acid(self):
        self.assertIsNone(phase_record({'eluent.A.formic':.1},'A','report')['reported_pH'])
    def test_invalid_ph(self):
        for value in ['NaN','inf',-1,15]:
            self.assertIsNone(phase_record({'eluent.A.pH':value},'A','report')['reported_pH'])
    def test_inactive_reservoir_excluded(self):
        self.assertEqual(active_channels({'gradient_program':[{'A [%]':100,'B [%]':0}]}),['A'])
    def test_match_retains_salt_and_gradient(self):
        a={'metadata':{'eluent.A.pH':3,'eluent.A.nh4ac':10},'gradient_program':[]}
        b={'metadata':{'eluent.A.pH':4,'eluent.A.nh4ac':10},'gradient_program':[]}
        self.assertEqual(signature_without_ph(a),signature_without_ph(b))
        b['metadata']['eluent.A.nh4ac']=20
        self.assertNotEqual(signature_without_ph(a),signature_without_ph(b))

if __name__=='__main__':unittest.main()
