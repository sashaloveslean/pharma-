import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_mcmrt_errors import audit


class ErrorAuditTests(unittest.TestCase):
    def test_comparison_checks_fold_data_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'old',Path(directory)/'new'
            a.mkdir();b.mkdir()
            plan={k:'fixed' for k in ('data_sha256','systems_sha256','splits','params','feature_grid_min')}
            old={'scenario':'unseen_method','fold':'0','observation_id':'a','system_id':'m',
                 'B_modifier':'methanol','cycle_min':10,'observed_tr_min':5,
                 'predicted_tr_min':9,'absolute_error_min':4}
            new={**old,'predicted_tr_min':6,'absolute_error_min':1}
            for folder,row in [(a,old),(b,new)]:
                (folder/'plan.json').write_text(json.dumps(plan))
                (folder/'predictions.jsonl').write_text(json.dumps(row)+'\n')
            result=next(x for x in audit(a,b) if x['group_by']=='all')
            self.assertEqual(result['change_mae_min'],-3)
            new['observed_tr_min']=6
            (b/'predictions.jsonl').write_text(json.dumps(new)+'\n')
            with self.assertRaises(ValueError):audit(a,b)
            (b/'predictions.jsonl').write_text(json.dumps(old)+'\n')
            (b/'plan.json').write_text(json.dumps({**plan,'splits':'changed'}))
            with self.assertRaises(ValueError):audit(a,b)
