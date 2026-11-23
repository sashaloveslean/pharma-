import sys
from pathlib import Path
import unittest
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from design_rnd_gradients import program, assess, choose, generate
from propose_pair_method import SPEC

class RndGradientTests(unittest.TestCase):
    def test_balances_and_preserved_tail_durations(self):
        for start in (10,15,20):
            p=program(start,1,50,6,12)
            self.assertTrue(all(x['A [%]']+x['B [%]']==100 for x in p))
            self.assertTrue(all(b['t [min]']>a['t [min]'] for a,b in zip(p,p[1:])))
            self.assertEqual(p[4]['t [min]']-p[3]['t [min]'],8)
            self.assertAlmostEqual(p[-1]['t [min]']-p[-2]['t [min]'],4.9)
            self.assertEqual(p[-1]['B [%]'],start)
            self.assertTrue(all(x['A [%]']<=90 for x in p))
    def test_invalid_design(self):
        with self.assertRaises(ValueError):program(5,1,50,6,12)
        with self.assertRaises(ValueError):program(10,7,50,6,12)
    def test_invalid_predictions_cannot_win(self):
        for tfv in (0,float('nan'),18):
            r=assess({'tenofovir':tfv,'emtricitabine':5},17)
            self.assertFalse(r['eligible_for_heuristic_ranking'])
            self.assertIsNone(r['gap_per_analytical_min'])
    def test_generator_does_not_mutate_source(self):
        m={'column.name':SPEC['name'],'column.length':100,'column.id':2.1,'column.particle.size':2.2,
           'column.temperature':30,'column.flowrate':.2,'eluent.A.pH':2.6,'eluent.A.h2o':100}
        system={'metadata':m};before=deepcopy(system);rows=generate(system)
        self.assertEqual(system,before);self.assertEqual(len(rows),162)
        self.assertEqual(len({r['candidate_id'] for r in rows}),162)
    def test_selected_designs_are_diverse(self):
        rows=[]
        for i in range(5):
            rows.append(dict(candidate_id=str(i),parameters={'start_B':10+i,'hold_min':i,'end_min':12},
                eligible_for_heuristic_ranking=True,gap_per_analytical_min=i/12,predicted_gap_min=i))
        selected=choose(rows)
        self.assertEqual(len(selected),3)
        self.assertEqual(len({r['candidate_id'] for r in selected}),3)
