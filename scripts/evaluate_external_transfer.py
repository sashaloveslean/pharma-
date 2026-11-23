"""Fixed external-source comparison of absolute and analytical-scaled RT models.

Only MCMRT trains the models. RepoRT sources are external to this experiment,
not claimed to be globally untouched throughout the project's earlier research.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
from rdkit import Chem, rdBase

from open_retention import load_observations
from evaluate_mcmrt_transfer import feature, analytical_scale, PARAMS, GRID, digest

SELECTED=['report:0390','report:0391','report:0400','report:0401','report:0402','report:0403']


def number(value):
    try:r=float(value)
    except (TypeError,ValueError):raise ValueError('missing_numeric_metadata')
    if not math.isfinite(r):raise ValueError('nonfinite_metadata')
    return r


def adapt_system(system):
    """No inferred concentrations, pH, t0 or instrument timings."""
    result=deepcopy(system);old=system['metadata'];meta={}
    if not system.get('source_info') or system['source_info'][0].get('method.type')!='RP':raise ValueError('not_RP')
    for k in ('column.length','column.id','column.particle.size','column.temperature','column.flowrate'):
        meta[k]=number(old.get(k))
        if meta[k]<=0:raise ValueError('invalid_geometry_temperature_or_flow')
    meta['column.name']=old['column.name'] # Preserve source product label; no post-hoc alias tuning.
    allowed={'h2o','meoh','acn','formic','nh4form','nh4ac','pH'}
    for ch in 'AB':
        for key in ('h2o','meoh','acn','formic','nh4form','nh4ac'):
            v=number(old.get(f'eluent.{ch}.{key}'))
            if v<0:raise ValueError('negative_concentration')
            meta[f'eluent.{ch}.{key}']=v
            if key in ('formic','nh4form','nh4ac') and v:
                required='%' if key=='formic' else 'mM'
                if old.get(f'eluent.{ch}.{key}.unit')!=required:raise ValueError('unsupported_concentration_unit')
        if abs(sum(meta[f'eluent.{ch}.{k}'] for k in ('h2o','meoh','acn'))-100)>1e-6:
            raise ValueError('unsupported_solvent_balance')
        for key,value in old.items():
            if key.startswith(f'eluent.{ch}.') and not key.endswith('.unit') and key.split('.')[-1] not in allowed:
                if value not in ('',None) and number(value)!=0:raise ValueError('unrepresented_additive_or_solvent')
    program=[]
    for p in system['gradient_program']:
        v={k:number(p.get(k)) for k in ('t [min]','A [%]','B [%]','flow rate [ml/min]')}
        if any(number(p.get(f'{ch} [%]',0))!=0 for ch in 'CD'):raise ValueError('nonbinary_program')
        if abs(v['A [%]']+v['B [%]']-100)>1e-6 or not 0<=v['B [%]']<=100:raise ValueError('invalid_program_composition')
        if v['flow rate [ml/min]']<=0:raise ValueError('invalid_flow')
        program.append(v)
    if len(program)<2 or program[0]['t [min]']!=0:raise ValueError('incomplete_program')
    if any(b['t [min]']<a['t [min]'] for a,b in zip(program,program[1:])):raise ValueError('decreasing_time')
    if not math.isclose(program[0]['flow rate [ml/min]'],meta['column.flowrate']):raise ValueError('metadata_program_flow_conflict')
    # Equal adjacent times encode a source-specified step, not invented elapsed time.
    # np.interp uses the right-side value exactly at the step, and the left value before.
    result.update(metadata=meta,gradient_program=program)
    analytical_scale(result)
    return result


def metrics(records):
    if not records:return {'n':0}
    return {'n':len(records),
        'absolute_mae_min':float(np.mean([abs(r['absolute_tr_min']-r['observed_tr_min']) for r in records])),
        'scaled_mae_min':float(np.mean([abs(r['scaled_tr_min']-r['observed_tr_min']) for r in records])),
        'scaled_p90_error_min':float(np.quantile([abs(r['scaled_tr_min']-r['observed_tr_min']) for r in records],.9)),
        'scaled_bias_min':float(np.mean([r['scaled_tr_min']-r['observed_tr_min'] for r in records]))}


def run():
    train_path=Path('data/processed/public_retention/mcmrt/observations.jsonl')
    external_path=Path('data/processed/public_retention/report/observations.jsonl')
    out=Path('reports/external_transfer');out.mkdir(parents=True,exist_ok=True)
    train=[r for r in load_observations(train_path) if r['eligible_for_rt_research']]
    systems=json.loads(external_path.with_name('systems.json').read_text())
    adapted={};excluded={}
    for sid in SELECTED:
        try:adapted[sid]=adapt_system(systems[sid])
        except ValueError as e:excluded[sid]=str(e)
    molecular_train={r['compound_id'].split('-')[0] for r in train}
    label_matches={(r['compound_id'],round(r['retention_time_min'],3)) for r in train}
    candidates=[];issues=Counter();duplicates=[];seen=set()
    with external_path.open() as f,rdBase.BlockLogs():
        for line in f:
            r=json.loads(line);sid=r['system_id']
            if sid not in adapted:continue
            mol=Chem.MolFromSmiles(r['smiles'])
            if mol is None or Chem.MolToInchiKey(mol)!=r['compound_id']:
                issues['structure_key_disagreement']+=1;continue
            rt=number(r['retention_time_min'])
            if rt<=0:issues['invalid_rt']+=1;continue
            # Count only once if two source libraries repeat the same molecule/method/time.
            method=digest({'metadata':adapted[sid]['metadata'],'program':adapted[sid]['gradient_program']})
            key=(method,r['compound_id'],round(rt,6))
            if key in seen:issues['duplicate_external_measurement_candidate']+=1;continue
            seen.add(key)
            if (r['compound_id'],round(rt,3)) in label_matches:
                duplicates.append(r['observation_id']);continue
            candidates.append({**adapted[sid],**r})
    plan={'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
          'train_systems_sha256':hashlib.sha256(train_path.with_name('systems.json').read_bytes()).hexdigest(),
          'external_sha256':hashlib.sha256(external_path.read_bytes()).hexdigest(),
          'external_systems_sha256':hashlib.sha256(external_path.with_name('systems.json').read_bytes()).hexdigest(),
          'selected_sources':SELECTED,'excluded_systems':excluded,'data_issues':dict(issues),
          'conservative_structure_rt_overlap_exclusions':duplicates,
          'train_ids':[r['observation_id'] for r in train],'test_ids':[r['observation_id'] for r in candidates],
          'params':PARAMS,'grid_min':GRID,'models':['absolute','analytical_scaled'],
          'scope':'Sources external to MCMRT-only training; not a globally untouched final test of the whole project.'}
    p=out/'plan.json'
    if p.exists() and digest(json.loads(p.read_text()))!=digest(plan):raise ValueError('Frozen external test changed')
    if not p.exists():p.write_text(json.dumps(plan,indent=2)+'\n')
    X=np.array([feature(r) for r in train],dtype=object)
    Xtest=np.array([feature(r) for r in candidates],dtype=object)
    y=np.array([r['retention_time_min'] for r in train])
    predictions={}
    for mode in ('absolute','scaled'):
        scales=np.array([analytical_scale(r) for r in train]) if mode=='scaled' else np.ones(len(train))
        test_scales=np.array([analytical_scale(r) for r in candidates]) if mode=='scaled' else np.ones(len(candidates))
        model=CatBoostRegressor(**PARAMS,cat_features=[X.shape[1]-1])
        model.fit(X,np.log1p(y/scales))
        predictions[mode]=np.maximum(0,np.expm1(model.predict(Xtest)))*test_scales
        model.save_model(str(out/f'research_{mode}.cbm'))
    records=[]
    for i,r in enumerate(candidates):
        records.append({'observation_id':r['observation_id'],'system_id':r['system_id'],
            'compound_id':r['compound_id'],'observed_tr_min':r['retention_time_min'],
            'absolute_tr_min':float(predictions['absolute'][i]),'scaled_tr_min':float(predictions['scaled'][i]),
            'molecule_seen_in_mcmrt':r['compound_id'].split('-')[0] in molecular_train,
            'outside_program_window':r['retention_time_min']>r['gradient_program'][-1]['t [min]']})
    summary={'status':'external_research_comparison_not_promoted','plan_sha256':digest(plan),
        'overall':metrics(records),'by_system':{sid:metrics([r for r in records if r['system_id']==sid]) for sid in adapted},
        'unseen_molecules':metrics([r for r in records if not r['molecule_seen_in_mcmrt']]),
        'seen_molecules':metrics([r for r in records if r['molecule_seen_in_mcmrt']]),
        'excluded_systems':excluded,'data_issues':dict(issues),'train_overlap_excluded':len(duplicates),
        'observations_outside_program_window':sum(r['outside_program_window'] for r in records),
        'production_approved':False,
        'target_diagnostics':{name:[r for r in records if r['compound_id']==key] for name,key in {
            'TFV':'SGOIRFVFHAKUTI-ZCFIWIBFSA-N','FTC':'XQSPYNMVSIKCOC-NTSWFWBYSA-N'}.items()},
        'limitations':['RepoRT has appeared in earlier project research; this is external to the current MCMRT models, not a global final holdout.',
            'Fixed configurations; no tuning on external errors.',
            'Source column labels preserved without post-hoc alias optimization.',
            'Conservative structure+rounded RT exclusions can remove coincidental matches; their IDs are retained.',
            'No independent widths/Rs or calibrated prediction intervals. No claim of TFV/FTC accuracy.']}
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    text=['# Внешняя проверка нормализации времени удерживания','',
          'Обе модели обучены только на MCMRT. Внешние источники и ограничения записаны до обучения; параметры моделей одинаковы.',
          'Это проверка переноса между источниками, не полностью нетронутая финальная проверка проекта: RepoRT использовался в предыдущих исследованиях.','',
          '| Группа | Наблюдений | Без нормализации, мин | С нормализацией, мин |','|---|---:|---:|---:|']
    for name,m in [('Все',summary['overall']),('Новые молекулы',summary['unseen_molecules']),*summary['by_system'].items()]:
        if m['n']:text.append(f"| {name} | {m['n']} | {m['absolute_mae_min']:.3f} | {m['scaled_mae_min']:.3f} |")
    text+=['','Исключённые методы: '+json.dumps(excluded,ensure_ascii=False),
           'Средние ошибки относятся к этим источникам и не определяют неопределённость отдельного прогноза.',
           'Прогнозы не обрезались до экспериментального окна, не калибровались по внешним ответам. Рабочая модель и выбранный градиент не заменены.',*summary['limitations']]
    text+=['','Проверка целевых веществ по полному InChIKey:']
    for name,rs in summary['target_diagnostics'].items():
        if not rs:text.append(f'{name}: измерений в этой внешней проверке нет.')
        for r in rs:
            text.append(f"{name}, {r['system_id']}: измерено {r['observed_tr_min']:.4f} мин, прогноз {r['scaled_tr_min']:.4f} мин. Это отдельный метод, не выбранный метод пары TFV/FTC.")
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
