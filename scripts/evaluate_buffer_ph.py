"""Exploratory pH ablation on audited RepoRT subset; never replaces production model."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from audit_buffer_ph import number
from train_open_retention import vector, metrics

ROOT=Path('data/processed/public_retention/report')
OUT=Path('reports/buffer_ph_evaluation')
PARAMS=dict(iterations=200,depth=5,learning_rate=.06,random_seed=42,verbose=False,thread_count=4,allow_writing_files=False)

def prepare():
    audited=[json.loads(l) for l in Path('reports/buffer_ph_audit/systems.jsonl').read_text().splitlines()]
    ids={r['system_id'] for r in audited if r['source']=='report' and r['rp'] and r['eligible'] and r['aqueous_ph_and_salt_units']}
    systems=json.loads((ROOT/'systems.json').read_text())
    rows=[];seen=set();duplicates=0
    for line in (ROOT/'observations.jsonl').open():
        r=json.loads(line)
        if r['system_id'] not in ids:continue
        if number(r.get('retention_time_min')) is None or r['retention_time_min']<=0:continue
        s=systems[r['system_id']]
        # Conservative exact-row deduplication, retaining differing outcomes.
        key=(r['system_id'],r['compound_id'],r['retention_time_min'])
        if key in seen:duplicates+=1;continue
        seen.add(key);r.update(s);rows.append(r)
    return rows,duplicates

def numeric_keys(rows):
    return sorted({k for r in rows for k,v in r['metadata'].items()
        if k!='id' and k!='column.t0' and not k.endswith('.pH') and number(v) is not None})

def run():
    rows,duplicates=prepare(); numeric=numeric_keys(rows)
    # Preserve units as categorical metadata rather than silently mixing percent/mM.
    unit_keys=sorted({k for r in rows for k in r['metadata'] if k.endswith('.unit')})
    base=[]
    for r in rows:
        v=vector(r,numeric)
        base.append(v[:-2]+[r['metadata'].get(k) or 'unspecified' for k in unit_keys]+v[-2:])
    X=np.asarray(base,dtype=object); cat=list(range(X.shape[1]-len(unit_keys)-2,X.shape[1]))
    ph=np.asarray([number(r['metadata'].get('eluent.A.pH')) for r in rows],dtype=float)
    if not np.isfinite(ph).all():raise ValueError('This experiment requires recorded A pH for every row')
    y=np.array([r['retention_time_min'] for r in rows])
    sources=np.array([r['source_info'][0]['url'].lower().rstrip('/') for r in rows])
    mols=np.array([r['compound_id'].split('-')[0] for r in rows])
    out=dict(rows=len(rows),exact_duplicates_removed=duplicates,methods=len({r['system_id'] for r in rows}),
        publications=len(set(sources)),molecules=len(set(mols)),pH_values=sorted(set(ph.tolist())),params=PARAMS,
        numeric_features=numeric,unit_features=unit_keys,target='log1p(tR_min)',production_updated=False,
        primary_source_check='Abstract confirms varying phases/pH. Exact recipes and titrants not independently verified against supplementary tables.',
        source_url='https://pubmed.ncbi.nlm.nih.gov/35139450/',
        limitations=['Exploratory imported metadata; not a validated pH response model.',
        'Only three publications; pH is confounded with publication/conditions.',
        'New-molecule test contains familiar methods; it does not measure new-method transfer.',
        'No TFV/FTC in this subset; no customer predictions or peak-shape forecast.',
        'Exact duplicates removed only within a system; unknown cross-source duplicates remain possible.'],scenarios={})
    predictions=[];splits=[]
    for scenario, splitter, groups in [('new_publication',LeaveOneGroupOut(),sources),('new_molecule',GroupKFold(5),mols)]:
        pred={name:np.zeros(len(rows)) for name in ['without_pH','with_pH']};folds=[]
        for fold,(train,test) in enumerate(splitter.split(X,y,groups)):
            assert not set(groups[train]) & set(groups[test])
            entry=dict(fold=fold,test_rows=len(test),held_out_groups=sorted(set(groups[test])),metrics={})
            splits.append(dict(scenario=scenario,fold=fold,train=[rows[i]['observation_id'] for i in train],test=[rows[i]['observation_id'] for i in test]))
            for name in pred:
                values=X if name=='without_pH' else np.column_stack([ph,X])
                cats=cat if name=='without_pH' else [i+1 for i in cat]
                model=CatBoostRegressor(**PARAMS,cat_features=cats)
                model.fit(values[train],np.log1p(y[train]))
                pred[name][test]=np.maximum(0,np.expm1(model.predict(values[test])))
                entry['metrics'][name]=metrics(y[test],pred[name][test])
            folds.append(entry)
        out['scenarios'][scenario]={name:metrics(y,p) for name,p in pred.items()}
        out['scenarios'][scenario]['folds']=folds
        out['scenarios'][scenario]['improvement_percent']=100*(1-np.mean(abs(y-pred['with_pH']))/np.mean(abs(y-pred['without_pH'])))
        for i,r in enumerate(rows):
            predictions.append(dict(scenario=scenario,observation_id=r['observation_id'],system_id=r['system_id'],compound_id=r['compound_id'],actual_min=float(y[i]),without_pH=float(pred['without_pH'][i]),with_pH=float(pred['with_pH'][i])))
        print(scenario,out['scenarios'][scenario]['improvement_percent'],flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    (OUT/'oof_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    lines=['# Проверка добавления pH в CatBoost','',f"Данные: {len(rows)} строк, {out['methods']} методов, {out['publications']} публикации. Удалено точных повторов: {duplicates}.",
        '', 'Одинаковые параметры, признаки и разбиения; единственное различие — добавление указанного pH резервуара A. Соли присутствуют в обоих вариантах. Цель — log1p(tR), ошибки ниже в минутах.', '',
        '| Проверка | MAE без pH | MAE с pH | Изменение ошибки |','|---|---:|---:|---:|']
    for name,s in out['scenarios'].items():
        lines.append(f"| {name} | {s['without_pH']['mae_min']:.3f} | {s['with_pH']['mae_min']:.3f} | {'снижение' if s['improvement_percent'] >= 0 else 'рост'} {abs(s['improvement_percent']):.1f}% |")
    lines+=['','Проверка new_publication полностью исключает публикацию из обучения. Проверка new_molecule исключает связность молекулы, но оставляет знакомые методы. Это разные задачи, их нельзя взаимозаменять.', '',
        'Первоисточник: [MultiConditionRT, 2022](https://pubmed.ncbi.nlm.nih.gov/35139450/). Аннотация подтверждает исследование разных pH и фаз; точные рецептуры четырёх пар и реагенты для титрования по приложению пока не подтверждены. Поэтому эксперимент исследовательский, а не аттестованная модель влияния pH.', '',
        'Только три публикации, pH связан с источником и условиями. TFV/FTC отсутствуют. Ни рабочая модель, ни рекомендации заказчику не изменены. Числа не подтверждают точность на приборе заказчика или способность прогнозировать Rs.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return out

if __name__=='__main__':run()
