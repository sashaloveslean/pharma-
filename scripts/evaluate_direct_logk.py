"""Compare direct molecular log(k) learning with LSS on frozen molecule folds.

Experimental isocratic log(k) only. One fixed column/recipe. No gradient or tR claim.
"""
import json
from collections import defaultdict
from pathlib import Path
import hashlib

from catboost import CatBoostRegressor
import numpy as np

from train_lss_parameters import molecule_folds
from train_open_retention import molecular


def run():
    root=Path('data/processed/lss_identity_audited')
    labels=[json.loads(l) for l in (root/'parameter_labels.jsonl').read_text().splitlines()]
    labels=[r for r in labels if r['identity']['smiles']]
    old=[json.loads(l) for l in Path('data/processed/lss_research/parameter_labels.jsonl').read_text().splitlines()]
    reference=[r['identity']['full_inchikey'].split('-')[0] for r in old if r['identity']['smiles']]
    groups=[r['identity']['full_inchikey'].split('-')[0] for r in labels]
    values=defaultdict(list)
    for line in Path('data/processed/lss_research/k_observations.jsonl').open():
        r=json.loads(line)
        values[(r['compound_number'],r['temperature_c'],r['phi'])].append(np.log10(r['k']))
    points=[]
    for i,r in enumerate(labels):
        for phi in sorted(r['calibration_phi']+r['validation_phi']):
            points.append({'series_index':i,'phi':phi,'y':float(np.mean(values[(r['compound_number'],r['temperature_c'],phi)])),
                           'X':molecular(r['identity']['smiles'])+[r['temperature_c'],phi],
                           'validation_composition':phi in r['validation_phi']})
    X=np.array([r['X'] for r in points]);y=np.array([r['y'] for r in points])
    # Fixed configuration, no selection on the evaluation predictions.
    params=dict(iterations=500,depth=5,learning_rate=.04,loss_function='RMSE',
                random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)
    predictions=[]
    for fold,train,test in molecule_folds(groups,reference):
        train_set=set(train);test_set=set(test)
        fit=[i for i,r in enumerate(points) if r['series_index'] in train_set]
        checks=[i for i,r in enumerate(points) if r['series_index'] in test_set and r['validation_composition']]
        model=CatBoostRegressor(**params);model.fit(X[fit],y[fit])
        for idx,pred in zip(checks,model.predict(X[checks])):
            point=points[idx];r=labels[point['series_index']]
            predictions.append({'fold':fold,'molecular_group':groups[point['series_index']],
                'series_id':r['series_id'],'name':r['name'],'phi':point['phi'],
                'observed_log10_k':point['y'],'predicted_log10_k':float(pred),
                'absolute_error_log10_k':abs(float(pred)-point['y'])})
    prior=[json.loads(l) for l in Path('reports/lss_parameter_learning_pubchem/predictions.jsonl').read_text().splitlines()]
    key=lambda r:(r['fold'],r['molecular_group'],r['series_id'],r['phi'])
    previous={key(r):r for r in prior}
    if {key(r) for r in predictions}!=set(previous):raise ValueError('Evaluation cases differ from frozen comparator')
    for r in predictions:r['lss_absolute_error_log10_k']=previous[key(r)]['absolute_error_log10_k']
    summary={'status':'research_comparison_not_promoted','cases':len(predictions),
             'mae_log10_k':float(np.mean([r['absolute_error_log10_k'] for r in predictions])),
             'lss_mae_log10_k':float(np.mean([r['lss_absolute_error_log10_k'] for r in predictions])),
             'training_compositions':'all observed compositions of training molecules; no test molecule measurements',
             'params':params,'dataset_sha256':hashlib.sha256((root/'parameter_labels.jsonl').read_bytes()).hexdigest(),
             'production_approved':False,'by_molecule':{},
             'limitations':['One fixed BEH C18/ACN system; no methanol or new-column validation.',
                            'Research folds already inspected; not a new final holdout.',
                            'Direct model does not impose monotonicity, and is not a calibrated gradient simulator.',
                            'Provisional PubChem name resolution; source substance identity remains unverified.']}
    for name in sorted({r['name'] for r in predictions}):
        subset=[r for r in predictions if r['name']==name]
        summary['by_molecule'][name]={'direct_mae':float(np.mean([r['absolute_error_log10_k'] for r in subset])),
                                    'lss_mae':float(np.mean([r['lss_absolute_error_log10_k'] for r in subset]))}
    out=Path('reports/direct_logk_comparison');out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    model=CatBoostRegressor(**params);model.fit(X,y);model.save_model(str(out/'research_direct_logk.cbm'))
    text=['# Прямой прогноз log(k) и прогноз через параметры LSS','',
          f"На {len(predictions)} одинаковых проверочных случаях: прямая модель MAE {summary['mae_log10_k']:.3f}; LSS-модель {summary['lss_mae_log10_k']:.3f}.",
          'Все температуры и составы тестируемой молекулы исключены из обучения. Использованы все составы обучающих молекул, поэтому сравнение отражает весь новый способ обучения, а не только архитектуру.', '',
          '| Вещество | Прямая модель | LSS |','|---|---:|---:|']
    for name in ['caffeine','ketoprofen','ibuprofen','methylparaben']:
        if name in summary['by_molecule']:
            r=summary['by_molecule'][name];text.append(f"| {name} | {r['direct_mae']:.3f} | {r['lss_mae']:.3f} |")
    text+=['','Это исследовательский результат для BEH C18/ACN. Автоматической замены рабочей модели нет. Выбранный градиент TFV/FTC сохранён.','',*summary['limitations']]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='by_molecule'},indent=2))


if __name__=='__main__':run()
