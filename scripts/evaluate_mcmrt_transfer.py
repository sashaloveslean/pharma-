"""Frozen research tests: unseen methods, unseen columns, and column+molecule shift.

MCMRT-only training prevents external-source overlap from leaking into evaluation.
No target labels, author projections or calibrant classifications enter features.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from open_retention import load_observations
from train_open_retention import molecular

PARAMS=dict(iterations=300,depth=6,learning_rate=.05,loss_function='RMSE',random_seed=42,
            thread_count=4,verbose=False,allow_writing_files=False)
GRID=[0,1,2,3,5,10,15,20,30,45,60,100]


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()


def feature(row,version='base'):
    if version not in {'base','physical_early_v1'}:raise ValueError('Unknown feature version')
    m=row['metadata'];p=row['gradient_program'];times=[v['t [min]'] for v in p]
    result=molecular(row['smiles'])+[m[k] for k in ('column.length','column.id','column.particle.size','column.temperature')]
    for ch in 'AB':
        result += [m[f'eluent.{ch}.{k}'] for k in ('h2o','meoh','acn','formic','nh4form','nh4ac')]
    result += [times[-1]]
    for solvent in ('h2o','meoh','acn'):
        concentrations=[(v['A [%]']*m[f'eluent.A.{solvent}']+v['B [%]']*m[f'eluent.B.{solvent}'])/100 for v in p]
        result += list(np.interp(GRID,times,concentrations))
    result += list(np.interp(GRID,times,[v['flow rate [ml/min]'] for v in p]))
    if version=='physical_early_v1':
        from early_retention_features import physical_features
        result+=list(physical_features(row).values())
    return result+[m['column.name']]


def analytical_scale(row):
    """First time at maximum actual organic fraction, excluding wash/re-equilibration.

    A feature-derived time scale, not t0 or a physical retention factor.
    Only supported for an increasing RP gradient starting at time zero.
    """
    m=row['metadata'];p=row['gradient_program']
    fractions=[sum(v[f'{ch} [%]']*(m[f'eluent.{ch}.meoh']+m[f'eluent.{ch}.acn'])/100
                   for ch in 'AB') for v in p]
    maximum=max(fractions)
    if maximum<=fractions[0]:raise ValueError('Analytical scale requires an increasing organic gradient')
    index=next(i for i,value in enumerate(fractions) if value==maximum)
    value=float(p[index]['t [min]'])
    if not np.isfinite(value) or value<=0:raise ValueError('Invalid analytical time scale')
    return value


def splits(rows):
    systems=np.array([r['system_id'] for r in rows])
    columns=np.array([r['metadata']['column.name'] for r in rows])
    molecules=np.array([r['compound_id'].split('-')[0] for r in rows])
    # Whole identical condition programs stay together, not merely their source IDs.
    methods=np.array([digest({'metadata':r['metadata'],'program':r['gradient_program']}) for r in rows])
    for fold,(fit,test) in enumerate(GroupKFold(5).split(rows,groups=methods)):
        yield 'unseen_method',str(fold),fit,test
    for fold,(fit,test) in enumerate(LeaveOneGroupOut().split(rows,groups=columns)):
        name=columns[test[0]]
        yield 'unseen_column_known_molecules',name,fit,test
        # Global molecular exclusion: no measurement of test molecules in any train column.
        held={g for g in set(molecules) if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==fold%5}
        joint_fit=np.array([i for i in fit if molecules[i] not in held],dtype=int)
        joint_test=np.array([i for i in test if molecules[i] in held],dtype=int)
        if len(joint_fit) and len(joint_test):yield 'unseen_column_and_molecules',name,joint_fit,joint_test


def compare_overlap(rows,report_path):
    index=defaultdict(list)
    for r in rows:index[(r['compound_id'],round(r['retention_time_min'],3))].append(r['system_id'])
    matches=Counter()
    for line in report_path.open():
        r=json.loads(line)
        for sid in index.get((r['compound_id'],round(r['retention_time_min'],3)),[]):
            matches[(sid,r['system_id'])]+=1
    return [{'mcmrt_system':a,'report_system':b,'matching_structure_rt_rows':n}
            for (a,b),n in matches.most_common(30)]


def run(data,out,target_mode='absolute',feature_version='base'):
    if target_mode not in {'absolute','analytical_scaled','analytical_log'}:raise ValueError('Unknown target mode')
    rows=[r for r in load_observations(data) if r['eligible_for_rt_research']]
    split_list=list(splits(rows))
    plan={'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
          'systems_sha256':hashlib.sha256(data.with_name('systems.json').read_bytes()).hexdigest(),
          'params':PARAMS,'feature_grid_min':GRID,
          'splits':[{'scenario':s,'fold':f,'training_ids':[rows[i]['observation_id'] for i in fit],
                     'test_ids':[rows[i]['observation_id'] for i in test]} for s,f,fit,test in split_list],
          'production_approved':False,'training_sources':['MCMRT supplement only']}
    if target_mode!='absolute':
        plan['target_mode']=('log' if target_mode=='analytical_log' else 'log1p')+'(tR / time_to_first_maximum_organic)'
        plan['rationale']='Research revision after error audit; original folds are development evidence, not a fresh final test.'
    if target_mode=='analytical_log':
        plan['early_exit_gate']={'early_definition':'observed tR / analytical scale <= 0.10; not a t0 criterion',
            'scenario':'unseen_column_and_molecules','require_early_mae_improvement':True,
            'maximum_overall_mae_ratio':1.05,'selection_data':'MCMRT only; no external FTC labels'}
    if feature_version!='base':
        from early_retention_features import physical_features
        plan['feature_version']=feature_version
        plan['extra_features']=list(physical_features(rows[0]))
        plan['research_gate']={'max_overall_mae_ratio_each_scenario':1.05,
            'joint_early_mae_must_improve':True,'early_fraction':.1,
            'reference':'reports/mcmrt_transfer_scaled','no_external_selection':True}
    out.mkdir(parents=True,exist_ok=True)
    plan_path=out/'plan.json'
    if plan_path.exists():
        if digest(json.loads(plan_path.read_text()))!=digest(plan):raise ValueError('Frozen evaluation plan changed; choose a new output directory')
    else:plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    X=np.asarray([feature(r,feature_version) for r in rows],dtype=object)
    y=np.array([r['retention_time_min'] for r in rows]);predictions=[];metrics=[]
    scales=np.array([analytical_scale(r) for r in rows]) if target_mode!='absolute' else np.ones(len(rows))
    for scenario,fold,fit,test in split_list:
        target=encode_target(y[fit],scales[fit],target_mode)
        model=CatBoostRegressor(**PARAMS,cat_features=[X.shape[1]-1]);model.fit(X[fit],target)
        pred=decode_target(model.predict(X[test]),scales[test],target_mode)
        ratios=defaultdict(list)
        for i in fit:ratios[rows[i]['compound_id'].split('-')[0]].append(y[i]/rows[i]['gradient_program'][-1]['t [min]'])
        fallback=float(np.median([v for vs in ratios.values() for v in vs]))
        baseline=np.array([np.median(ratios.get(rows[i]['compound_id'].split('-')[0],[fallback]))*
                           rows[i]['gradient_program'][-1]['t [min]'] for i in test])
        item={'scenario':scenario,'fold':fold,'train_rows':len(fit),'test_rows':len(test),
              'mae_min':float(np.mean(abs(pred-y[test]))),'baseline_mae_min':float(np.mean(abs(baseline-y[test]))),
              'median_absolute_error_min':float(np.median(abs(pred-y[test])))}
        metrics.append(item);print(json.dumps(item),flush=True)
        for i,estimate,base in zip(test,pred,baseline):
            predictions.append({'scenario':scenario,'fold':fold,'observation_id':rows[i]['observation_id'],
                'system_id':rows[i]['system_id'],'compound_id':rows[i]['compound_id'],
                'observed_tr_min':float(y[i]),'predicted_tr_min':float(estimate),'baseline_tr_min':float(base),
                'B_modifier':'methanol' if rows[i]['metadata']['eluent.B.meoh']==100 else 'acetonitrile',
                'absolute_error_min':float(abs(y[i]-estimate)),'cycle_min':rows[i]['gradient_program'][-1]['t [min]']})
    aggregate={}
    for scenario in sorted({r['scenario'] for r in predictions}):
        rs=[r for r in predictions if r['scenario']==scenario]
        aggregate[scenario]={'test_rows':len(rs),'mae_min':float(np.mean([r['absolute_error_min'] for r in rs])),
            'baseline_mae_min':float(np.mean([abs(r['observed_tr_min']-r['baseline_tr_min']) for r in rs])),
            'macro_fold_mae_min':float(np.mean([r['mae_min'] for r in metrics if r['scenario']==scenario])),
            'by_modifier':{mod:float(np.mean([r['absolute_error_min'] for r in rs if r['B_modifier']==mod]))
                           for mod in sorted({r['B_modifier'] for r in rs})}}
    overlap=compare_overlap(rows,Path('data/processed/public_retention/report/observations.jsonl'))
    report={'status':'research_evaluation_not_promoted','scenarios':aggregate,'folds':metrics,
            'target_mode':target_mode,
            'feature_version':feature_version,
            'overlap_candidates_with_report':overlap,'plan_sha256':digest(plan),
            'baseline':'Median training tR/cycle for the same molecular connectivity; global median for unseen molecules.',
            'limitations':['Single laboratory/publication; held-out column is not held-out laboratory.',
                'No TFV or FTC observations in this dataset.',
                'pH, measured void time and dwell volume absent in the downloaded supplementary methods.',
                'No widths, pressure or resolution forecast. RT classification and author predictions excluded.',
                'Only method CMs without metadata conflicts and molecules with agreeing SMILES/InChI were used.',
                'Matching structure+rounded RT across sources is a duplicate candidate, not proof; no cross-source pooling.']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    lines=['# Проверка переноса на MCMRT','',
           'Обучение и проверка только внутри нового набора, без смешивания с RepoRT. План и параметры зафиксированы до запуска.',
           '', '| Проверка | Строк проверки | MAE, мин | Простой ориентир, мин |','|---|---:|---:|---:|']
    for scenario,m in aggregate.items():lines.append(f"| {scenario} | {m['test_rows']} | {m['mae_min']:.3f} | {m['baseline_mae_min']:.3f} |")
    lines+=['','Сценарии: новая программа; новая колонка с преимущественно знакомыми молекулами; одновременно новая колонка и новые молекулы.',
            f'Режим целевой переменной: {target_mode}. Масштабирование, если включено, не является расчётом k или t₀.',
            'В последнем сценарии все измерения тестируемых молекул удалены из обучения на всех колонках. В первом сценарии одинаковые полные условия группируются вместе.',
            'Все значения в минутах. У методов разные длительности; общий MAE следует читать вместе с результатами отдельных колонок и растворителей.',
            'TFV/FTC в данных нет: это проверка переноса подхода, не подтверждение выбранного метода заказчика.', '', *report['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def encode_target(y,scales,mode):
    y=np.asarray(y);scales=np.asarray(scales)
    if not np.all(np.isfinite(y)) or not np.all(y>0) or not np.all(np.isfinite(scales)) or not np.all(scales>0):
        raise ValueError('Finite positive measured times and scales required')
    return np.log(y/scales) if mode=='analytical_log' else np.log1p(y/scales)


def decode_target(values,scales,mode):
    return (np.exp(values) if mode=='analytical_log' else np.maximum(0,np.expm1(values)))*scales


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed/public_retention/mcmrt/observations.jsonl'))
    p.add_argument('--out',type=Path,default=Path('reports/mcmrt_transfer'))
    p.add_argument('--target-mode',choices=['absolute','analytical_scaled','analytical_log'],default='absolute')
    p.add_argument('--feature-version',choices=['base','physical_early_v1'],default='base')
    a=p.parse_args();print(json.dumps(run(a.data,a.out,a.target_mode,a.feature_version)['scenarios'],indent=2))
