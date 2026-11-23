"""Research-only prediction of paired tR changes, not a causal pH or customer model."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from audit_buffer_ph import signature_without_ph
from train_open_retention import vector, metrics
from evaluate_buffer_ph import numeric_keys

ROOT=Path('data/processed/public_retention/report')
OUT=Path('reports/ph_delta_evaluation')
PARAMS=dict(iterations=200,depth=4,learning_rate=.04,random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)

def unique_outcome(records):
    # Multiple distinct outcomes are not silently averaged or treated as replicates.
    values={float(r['retention_time_min']) for r in records}
    structures={r['smiles'] for r in records}
    return records[0] if len(values)==1 and len(structures)==1 else None

def prepare():
    systems=json.loads((ROOT/'systems.json').read_text())
    pairs=json.loads(Path('reports/buffer_ph_audit/audit.json').read_text())['strict_ph_comparison_groups']
    index=defaultdict(lambda:defaultdict(list))
    for line in (ROOT/'observations.jsonl').open():
        r=json.loads(line)
        if r.get('retention_time_min',0)>0:index[r['system_id']][r['compound_id']].append(r)
    result=[];skipped=[]
    for pair in pairs:
        if pair['source']!='report':continue
        low,high=sorted(pair['systems'],key=lambda s:float(systems[s]['metadata']['eluent.A.pH']))
        assert signature_without_ph(systems[low])==signature_without_ph(systems[high])
        for cid in sorted(index[low].keys() & index[high].keys()):
            a=unique_outcome(index[low][cid]);b=unique_outcome(index[high][cid])
            if a is None or b is None or a['smiles']!=b['smiles']:
                skipped.append([low,high,cid]);continue
            row=dict(a);row.update(systems[low])
            row.update(pair_id=low+'__'+high,low_system=low,high_system=high,
                low_observation_id=a['observation_id'],high_observation_id=b['observation_id'],
                tr_low=float(a['retention_time_min']),tr_high=float(b['retention_time_min']),
                delta_min=float(b['retention_time_min'])-float(a['retention_time_min']),
                ph_low=float(systems[low]['metadata']['eluent.A.pH']),ph_high=float(systems[high]['metadata']['eluent.A.pH']))
            result.append(row)
    return result,skipped

def run():
    rows,skipped=prepare();numeric=numeric_keys(rows)
    X=np.asarray([vector(r,numeric) for r in rows],dtype=object)
    y=np.asarray([r['delta_min'] for r in rows]);groups=np.asarray([r['compound_id'].split('-')[0] for r in rows])
    predictions={k:np.zeros(len(rows)) for k in ('zero_change','train_median','catboost')}
    folds=[];split_records=[]
    for fold,(train,test) in enumerate(GroupKFold(5).split(X,y,groups)):
        assert not set(groups[train]) & set(groups[test])
        model=CatBoostRegressor(**PARAMS,cat_features=[X.shape[1]-2,X.shape[1]-1])
        model.fit(X[train],y[train]);predictions['catboost'][test]=model.predict(X[test])
        predictions['train_median'][test]=np.median(y[train])
        folds.append(dict(fold=fold,test_rows=len(test),metrics={k:metrics(y[test],p[test]) for k,p in predictions.items()}))
        split_records.append(dict(fold=fold,train_compound_groups=sorted(set(groups[train])),test_compound_groups=sorted(set(groups[test]))))
    summary={k:metrics(y,p) for k,p in predictions.items()}
    per_pair={pair:{k:metrics(y[mask],p[mask]) for k,p in predictions.items()}
              for pair in sorted({r['pair_id'] for r in rows})
              if (mask:=np.array([r['pair_id']==pair for r in rows])).any()}
    # Paired molecule-cluster bootstrap: descriptive CI, not independent external validation.
    rng=np.random.default_rng(42);ids=sorted(set(groups));boot=[]
    errors=abs(y-predictions['zero_change'])-abs(y-predictions['catboost'])
    for _ in range(1000):
        sampled=rng.choice(ids,len(ids),replace=True)
        boot.append(float(np.mean(np.concatenate([errors[groups==g] for g in sampled]))))
    result=dict(rows=len(rows),molecule_connectivity_groups=len(ids),pairs=len(per_pair),
        skipped_ambiguous=skipped,params=PARAMS,numeric_features=numeric,target='tR_pH8_minus_tR_pH5_min',
        response_features_used=False,split='5-fold molecule-connectivity-disjoint; familiar method pairs',
        summary=summary,per_pair=per_pair,folds=folds,
        mae_gain_vs_zero_cluster_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),
        production_updated=False,customer_prediction_supported=False,
        limitations=['Single publication, four method pairs: no external validation.',
        'Only transition 5 to 8: no learned continuous pH response or interpolation to pH 3.',
        'Recipe/titrant details not verified against original supplement; recorded metadata match only.',
        'No TFV/FTC evaluation; predicts a difference, not absolute tR, peak shape or Rs.',
        'Baseline measured tR is not a feature. Absolute prediction would require separately validated baseline.'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'splits.json').write_text(json.dumps(split_records,indent=2)+'\n')
    outputs=[]
    for i,r in enumerate(rows):
        outputs.append({k:r[k] for k in ['pair_id','compound_id','name','low_observation_id','high_observation_id','tr_low','tr_high','delta_min','ph_low','ph_high']}|{k:float(p[i]) for k,p in predictions.items()})
    (OUT/'paired_oof.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in outputs))
    lines=['# Прогноз изменения удерживания между pH 5 и 8','',
        f"{len(rows)} парных наблюдений, {len(ids)} групп связности молекул, {len(per_pair)} пары методов одной публикации. Неоднозначных пар исключено: {len(skipped)}.",'',
        'Цель: tR(pH 8) − tR(pH 5). Ни одно измеренное tR не входит в признаки. Все условия одной тестовой молекулы исключены из обучения вместе. Методы при этом знакомы модели.', '',
        '| Вариант | MAE изменения tR, мин |','|---|---:|']
    for k,v in summary.items():lines.append(f"| {k} | {v['mae_min']:.3f} |")
    lines+=['',f"Описательный 95% bootstrap-интервал выигрыша CatBoost относительно нулевого изменения: {result['mae_gain_vs_zero_cluster_bootstrap_95']}. Пересэмплирование по молекулам; это не независимая внешняя проверка.",'',
        'Первоисточник: https://doi.org/10.1016/j.chroma.2022.462867. Доступ к полному тексту через издателя вернул HTTP 403; точные составы и титранты по приложению не подтверждены. Совпадение остальных полей RepoRT не доказывает изолированный причинный эффект pH.', '',
        'Это отдельный исследовательский тест на знакомых методах одной статьи. Он не даёт обоснованного прогноза для TFV/FTC, pH 3–4, других буферов или прибора заказчика. Рабочая модель и рекомендации не изменены.']
    lines += ['', 'Решение: не внедрять в рабочий предиктор. В этом тесте CatBoost не превзошёл нулевое изменение. Подбор гиперпараметров по этим же тестовым результатам не проводился; требуется подтверждение рецептур и дополнительная независимая серия.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['rows','molecule_connectivity_groups','summary','mae_gain_vs_zero_cluster_bootstrap_95']},indent=2))
    return result

if __name__=='__main__':run()
