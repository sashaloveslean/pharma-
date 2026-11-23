"""Fixed ridge residual adapter using gradient profiles; identical held-out cases."""
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_mcmrt_transfer import analytical_scale, digest
from evaluate_local_gradient_model import SCENARIO

ALPHA=0.1
ABS_GRID=np.linspace(0,100,41)
REL_GRID=np.linspace(0,1,21)


def shape_features(system):
    m=system['metadata'];p=system['gradient_program'];duration=analytical_scale(system)
    stop=next(i for i,x in enumerate(p) if x['t [min]']==duration)
    p=p[:stop+1] # Wash/return are not causal predictors of earlier elution.
    times=np.array([x['t [min]'] for x in p],float)
    organic=np.array([sum(x[f'{ch} [%]']*(float(m[f'eluent.{ch}.meoh'])+float(m[f'eluent.{ch}.acn']))/10000 for ch in 'AB') for x in p])
    if not np.all(np.isfinite(organic)):raise ValueError('Invalid phase composition')
    # Fixed scaling, no test labels or test-fitted standardization.
    return np.r_[np.log(duration),np.interp(ABS_GRID,times,organic)/np.sqrt(len(ABS_GRID)),
                 np.interp(REL_GRID*duration,times,organic)/np.sqrt(len(REL_GRID))]


def adapt(X,y,test):
    X=np.asarray(X,float);y=np.asarray(y,float);test=np.asarray(test,float)
    if len(X)<2 or len(X)!=len(y) or not all(np.all(np.isfinite(v)) for v in (X,y,test)):
        raise ValueError('Invalid calibration data')
    center=X.mean(axis=0);z=X-center
    # Dual ridge: small calibration panel, unpenalized intercept.
    weights=np.linalg.solve(z@z.T+ALPHA*np.eye(len(X)),y-y.mean())
    return float(y.mean()+(test-center)@z.T@weights)


def metrics(rows):
    return {'n':len(rows),**{k+'_mae_min':float(np.mean([abs(r[k]-r['observed_tr_min']) for r in rows]))
            for k in ('catboost','constant_offset','local_slope','shape_model')},
            'shape_clipped_to_zero':sum(r['shape_unclipped']<0 for r in rows)}


def run():
    prior=Path('reports/local_gradient_model');base=Path('reports/mcmrt_transfer_scaled')
    pp=json.loads((prior/'plan.json').read_text());bp=json.loads((base/'plan.json').read_text())
    sp=Path('data/processed/public_retention/mcmrt/systems.json');systems=json.loads(sp.read_text())
    if hashlib.sha256(sp.read_bytes()).hexdigest()!=bp['systems_sha256'] or digest(bp)!=pp['base_plan_sha256']:
        raise ValueError('Source metadata or base plan changed')
    raw=(base/'predictions.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pp['prediction_sha256']:raise ValueError('Base predictions changed')
    index={r['observation_id']:r for r in map(json.loads,raw.splitlines()) if r['scenario']==SCENARIO}
    old={r['test_id']:r for r in map(json.loads,(prior/'predictions.jsonl').read_text().splitlines())}
    if set(old)!={c['test_id'] for c in pp['cases']}:raise ValueError('Comparison cases differ')
    out=Path('reports/local_gradient_shape');out.mkdir(parents=True,exist_ok=True)
    plan={'parent_plan_sha256':digest(pp),'parent_predictions_sha256':hashlib.sha256((prior/'predictions.jsonl').read_bytes()).hexdigest(),
          'alpha':ALPHA,'absolute_grid_min':ABS_GRID.tolist(),'relative_grid':REL_GRID.tolist(),
          'features':'log analytical duration + actual organic profiles, each grid divided by sqrt(grid length); truncate at first maximum',
          'test_ids':[c['test_id'] for c in pp['cases']],'no_hyperparameter_search':True}
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen shape plan changed')
    if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
    results=[]
    for c in pp['cases']:
        if c['test_id'] in c['calibration_ids']:raise ValueError('Test label leakage')
        train=[index[i] for i in c['calibration_ids']];test=index[c['test_id']]
        X=[shape_features(systems[r['system_id']]) for r in train]
        y=[r['observed_tr_min']-r['predicted_tr_min'] for r in train]
        raw_pred=test['predicted_tr_min']+adapt(X,y,shape_features(systems[test['system_id']]))
        previous=old[c['test_id']]
        if previous['observed_tr_min']!=test['observed_tr_min']:raise ValueError('Test label changed')
        results.append({**previous,'shape_model':max(0.,raw_pred),'shape_unclipped':raw_pred})
    summary={'overall':metrics(results),'by_program':{sid:metrics([r for r in results if r['test_system']==sid]) for sid in sorted({r['test_system'] for r in results})},
             'production_approved':False,'target_pair_validated':False,
             'limitations':['Only 2–3 local programs per molecule; high dimensional shape is heavily regularized.',
                'Same 440 retrospective cases, no new blind validation or hyperparameter tuning.',
                'Known measured molecule on new program; not prediction for unmeasured TFV/FTC.',
                'Shape grids are approximations, not a mechanistic column simulator.',
                'No peak width, Rs or calibrated uncertainty. Negative predictions counted and clipped.']}
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    lines=['# Проверка признаков формы градиента','',
        'Те же 440 контрольных случаев; 2–3 локальные программы для настройки, отдельная программа для проверки.',
        'Модель поправки учитывает реальную долю органики на абсолютной и нормированной временных сетках до первого максимума. Поздняя промывка исключена из признаков поправки.', '',
        '| Программа | n | CatBoost | Постоянная поправка | По длительности | По форме |','|---|---:|---:|---:|---:|---:|']
    for label,m in [('Все',summary['overall']),*summary['by_program'].items()]:
        lines.append(f"| {label} | {m['n']} | {m['catboost_mae_min']:.3f} | {m['constant_offset_mae_min']:.3f} | {m['local_slope_mae_min']:.3f} | {m['shape_model_mae_min']:.3f} |")
    lines+=['','Все ошибки — MAE tR в минутах. Рабочая модель не заменена.',*summary['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
