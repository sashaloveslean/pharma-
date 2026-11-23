from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_mcmrt import dimensions, phase
from evaluate_mcmrt_transfer import splits, feature, digest, analytical_scale
from train_open_retention import molecular


def record(column=0,method=0,molecule=0):
    m={'column.name':f'Column {column}','column.length':100.,'column.id':2.1,
       'column.particle.size':2.,'column.temperature':30.}
    for ch,text in [('A','Water/methanol=90:10, v/v with 0.01% formic acid and 5 mM ammonium formate'),
                    ('B','Methanol with 0.01% formic acid and 5 mM ammonium formate')]:
        m.update({f'eluent.{ch}.{k}':v for k,v in phase(text).items()})
    p=[{'t [min]':0.,'A [%]':100.,'B [%]':0.,'flow rate [ml/min]':.3},
       {'t [min]':float(10+method),'A [%]':0.,'B [%]':100.,'flow rate [ml/min]':.3}]
    return {'system_id':f'{column}:{method}','compound_id':f'molecule{molecule}-key-N',
            'metadata':m,'gradient_program':p,'smiles':'CCO','retention_time_min':5.}


class MCMRTTests(unittest.TestCase):
    def test_analytical_scale_excludes_wash_and_return(self):
        row=record();base=analytical_scale(row)
        row['gradient_program'] += [
            {'t [min]':30.,'A [%]':0.,'B [%]':100.,'flow rate [ml/min]':.3},
            {'t [min]':31.,'A [%]':100.,'B [%]':0.,'flow rate [ml/min]':.3},
            {'t [min]':60.,'A [%]':100.,'B [%]':0.,'flow rate [ml/min]':.3}]
        self.assertEqual(analytical_scale(row),base)
        row['retention_time_min']=999
        self.assertEqual(analytical_scale(row),base)

    def test_time_scaling_and_isocratic_guard(self):
        row=record();base=analytical_scale(row)
        for point in row['gradient_program']:point['t [min]']*=3
        self.assertEqual(analytical_scale(row),3*base)
        for point in row['gradient_program']:point.update({'A [%]':100.,'B [%]':0.})
        with self.assertRaises(ValueError):analytical_scale(row)

    def test_mixed_column_dimension_order(self):
        self.assertEqual(dimensions('2.1 x 50 mm, 1.8 μm'),(50.,2.1,1.8))
        self.assertEqual(dimensions('100x2.1 mm, 2.2 μm'),(100.,2.1,2.2))
        with self.assertRaises(ValueError):dimensions('unknown')

    def test_buffer_units_and_actual_organic_fraction(self):
        p=phase('Water/methanol=90:10, v/v with 0.01% formic acid and 5 mM ammonium formate')
        self.assertEqual(p['meoh'],10.)
        self.assertEqual(p['formic'],.01)
        self.assertEqual(p['nh4form'],5.)
        row=record();values=feature(row)
        start=len(molecular(row['smiles']))+4+12+1
        self.assertEqual(values[start],90.)
        self.assertEqual(values[start+12],10.)  # B=0 still means 10% methanol, not 0%.
        self.assertEqual(values[start+24],0.)

    def test_response_and_author_classifications_never_enter_features(self):
        row=record();other=deepcopy(row)
        other.update(retention_time_min=99,calibrant='Calibrant',classification=13,source_prediction=100)
        self.assertEqual(feature(row),feature(other))

    def test_independent_method_column_and_joint_splits(self):
        rows=[record(c,m,i) for c in range(6) for m in range(2) for i in range(30)]
        methods=lambda ids:{digest({'metadata':rows[i]['metadata'],'program':rows[i]['gradient_program']}) for i in ids}
        columns=lambda ids:{rows[i]['metadata']['column.name'] for i in ids}
        molecules=lambda ids:{rows[i]['compound_id'].split('-')[0] for i in ids}
        for scenario,_,fit,test in splits(rows):
            self.assertFalse(set(fit)&set(test))
            if scenario=='unseen_method':self.assertFalse(methods(fit)&methods(test))
            else:self.assertFalse(columns(fit)&columns(test))
            if scenario=='unseen_column_and_molecules':self.assertFalse(molecules(fit)&molecules(test))
