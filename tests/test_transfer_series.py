import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_transfer_series import audit


class TransferSeriesTests(unittest.TestCase):
    def test_aliases_are_not_new_programs_and_missing_temperature_remains_blocker(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);systems={};rows=[]
            for sid,end in [('a',10),('b',10),('c',20),('d',30)]:
                systems[sid]={'metadata':{'column.name':'test','column.temperature':'','eluent.B.meoh':'100'},
                    'source_info':[{'method.type':'RP','url':'https://example.org/source'}],
                    'gradient_program':[{'t [min]':0,'B [%]':0},{'t [min]':end,'B [%]':100}]}
                rows.append({'system_id':sid,'compound_id':'EXACT-IDENTITY-N'})
            (root/'systems.json').write_text(json.dumps(systems))
            (root/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            groups=audit(root)['groups']
            self.assertEqual(len(groups),1)
            self.assertEqual(groups[0]['distinct_full_programs'],3)
            self.assertEqual(groups[0]['molecules_with_at_least_three_programs'],1)
            self.assertIn('missing_temperature',groups[0]['blockers'])
            self.assertFalse(groups[0]['training_ready'])

    def test_different_columns_are_never_merged(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);systems={}
            for i in range(3):
                systems[str(i)]={'metadata':{'column.name':str(i)},
                    'source_info':[{'method.type':'RP','url':'https://example.org/source'}],
                    'gradient_program':[{'t [min]':0},{'t [min]':10+i}]}
            (root/'systems.json').write_text(json.dumps(systems))
            (root/'observations.jsonl').write_text('')
            self.assertEqual(audit(root)['groups'],[])
