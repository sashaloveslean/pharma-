from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_external_transfer import adapt_system, metrics


def source():
    meta={'column.name':'Example C18','column.length':'100','column.id':'2.1',
          'column.particle.size':'1.7','column.temperature':'40','column.flowrate':'.3'}
    for ch in 'AB':
        for key in ('h2o','meoh','acn','formic','nh4form','nh4ac'):
            meta[f'eluent.{ch}.{key}']='0'
        meta[f'eluent.{ch}.formic']='0.1';meta[f'eluent.{ch}.formic.unit']='%'
    meta['eluent.A.h2o']='100';meta['eluent.B.acn']='100'
    program=[{'t [min]':t,'A [%]':100-b,'B [%]':b,'flow rate [ml/min]':.3}
             for t,b in [(0,5),(7,100),(8,100),(8,5),(9,5)]]
    return {'metadata':meta,'gradient_program':program,'source_info':[{'method.type':'RP'}]}


class ExternalTransferTests(unittest.TestCase):
    def test_explicit_zero_time_step_preserved_without_inventing_duration(self):
        original=source();r=adapt_system(original)
        self.assertEqual(r['gradient_program'][2]['t [min]'],r['gradient_program'][3]['t [min]'])
        self.assertEqual(original['metadata']['column.flowrate'],'.3')
        self.assertEqual(r['metadata']['column.flowrate'],.3)

    def test_unsupported_additives_and_flow_conflicts_excluded(self):
        r=source();r['metadata']['eluent.A.nh4oh']='6.5'
        with self.assertRaisesRegex(ValueError,'unrepresented'):adapt_system(r)
        r=source();r['metadata']['column.flowrate']='.35'
        with self.assertRaisesRegex(ValueError,'flow_conflict'):adapt_system(r)
        r=source();r['metadata']['eluent.A.formic.unit']='mM'
        with self.assertRaisesRegex(ValueError,'unit'):adapt_system(r)

    def test_missing_solvent_not_silently_zeroed(self):
        r=source();del r['metadata']['eluent.B.meoh']
        with self.assertRaises(ValueError):adapt_system(r)

    def test_metric_direction(self):
        m=metrics([{'absolute_tr_min':8,'scaled_tr_min':4,'observed_tr_min':5}])
        self.assertEqual(m['absolute_mae_min'],3)
        self.assertEqual(m['scaled_mae_min'],1)
        self.assertEqual(m['scaled_bias_min'],-1)
