"""One external diagnostic after a predeclared MCMRT-only screening criterion."""
import hashlib
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor

from open_retention import load_observations
from evaluate_mcmrt_transfer import feature,analytical_scale,encode_target,decode_target,PARAMS,digest
from evaluate_external_transfer import adapt_system


def run():
    gate_path=Path('reports/early_exit_audit/evaluation.json')
    gate=json.loads(gate_path.read_text())
    if not gate['passes_internal_research_gate']:raise ValueError('Candidate failed the internal research gate')
    reference=Path('reports/external_transfer')
    original=json.loads((reference/'plan.json').read_text())
    train_path=Path('data/processed/public_retention/mcmrt/observations.jsonl')
    ext_path=Path('data/processed/public_retention/report/observations.jsonl')
    for path,key in [(train_path,'train_sha256'),(train_path.with_name('systems.json'),'train_systems_sha256'),
                     (ext_path,'external_sha256'),(ext_path.with_name('systems.json'),'external_systems_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=original[key]:raise ValueError('Frozen data changed')
    train_by_id={r['observation_id']:r for r in load_observations(train_path)}
    train=[train_by_id[i] for i in original['train_ids']]
    systems=json.loads(ext_path.with_name('systems.json').read_text());by_id={};wanted=set(original['test_ids'])
    with ext_path.open() as f:
        for line in f:
            row=json.loads(line)
            if row['observation_id'] in wanted:by_id[row['observation_id']]={**adapt_system(systems[row['system_id']]),**row}
    test=[by_id[i] for i in original['test_ids']]
    out=Path('reports/early_exit_external');out.mkdir(parents=True,exist_ok=True)
    plan={'reference_plan_sha256':digest(original),'gate_sha256':hashlib.sha256(gate_path.read_bytes()).hexdigest(),
          'params':PARAMS,'target':'log(tR / analytical_scale)','test_ids':original['test_ids'],
          'external_status':'previously inspected external diagnostic, not a fresh final holdout'}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=plan:raise ValueError('Frozen candidate plan changed')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    X=np.array([feature(r) for r in train],dtype=object);Xt=np.array([feature(r) for r in test],dtype=object)
    scales=np.array([analytical_scale(r) for r in train]);ts=np.array([analytical_scale(r) for r in test])
    y=np.array([r['retention_time_min'] for r in train])
    model=CatBoostRegressor(**PARAMS,cat_features=[X.shape[1]-1])
    model.fit(X,encode_target(y,scales,'analytical_log'))
    pred=decode_target(model.predict(Xt),ts,'analytical_log')
    previous={r['observation_id']:r for r in [json.loads(l) for l in (reference/'predictions.jsonl').read_text().splitlines()]}
    records=[]
    for r,value,scale in zip(test,pred,ts):
        old=previous[r['observation_id']]
        if r['retention_time_min']!=old['observed_tr_min']:raise ValueError('Observed external label changed')
        records.append({**old,'candidate_tr_min':float(value),
                        'early_by_observed_time':bool(r['retention_time_min']/scale<=.1)})
    def metrics(rs):
        return {'n':len(rs),'previous_mae_min':float(np.mean([abs(r['scaled_tr_min']-r['observed_tr_min']) for r in rs])) if rs else None,
                'candidate_mae_min':float(np.mean([abs(r['candidate_tr_min']-r['observed_tr_min']) for r in rs])) if rs else None}
    result={'overall':metrics(records),'early':metrics([r for r in records if r['early_by_observed_time']]),
            'other':metrics([r for r in records if not r['early_by_observed_time']]),
            'by_system':{sid:metrics([r for r in records if r['system_id']==sid]) for sid in sorted({r['system_id'] for r in records})},
            'FTC':[r for r in records if r['compound_id']=='XQSPYNMVSIKCOC-NTSWFWBYSA-N'],
            'production_approved':False,'external_status':plan['external_status']}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    model.save_model(str(out/'research_log_ratio.cbm'))
    lines=['# Внешняя диагностика раннего выхода','',
           'Кандидат выбран по предварительному критерию MCMRT, внешние ответы не использовались для обучения или подбора параметров.',
           'Эта внешняя выборка уже изучалась: результаты являются повторной диагностикой, не новой слепой проверкой.','',
           '| Группа | N | Было MAE, мин | Стало MAE, мин |','|---|---:|---:|---:|']
    for name,m in [('Все',result['overall']),('Ранние',result['early']),('Остальные',result['other']),*result['by_system'].items()]:
        if m['n']:lines.append(f"| {name} | {m['n']} | {m['previous_mae_min']:.3f} | {m['candidate_mae_min']:.3f} |")
    for r in result['FTC']:lines.append(f"\nFTC: измерено {r['observed_tr_min']:.4f} мин, прежний прогноз {r['scaled_tr_min']:.4f}, новый {r['candidate_tr_min']:.4f}. Другой метод (BEH C18/ACN), не выбранная программа TFV/FTC.")
    lines+=['','Ранний выход определён по первым 10% аналитического времени, а не по измеренному t₀. Выделение групп по измеренному времени служит только диагностике, не переключает модель при прогнозе.',
            'Основная модель не заменена. Один пример FTC не подтверждает качество разделения смеси; TFV здесь отсутствует.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
