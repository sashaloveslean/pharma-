"""Leave-one-program-out adaptation for measured molecules on a fixed column."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_gradient_changes import method_pairs
from evaluate_mcmrt_transfer import analytical_scale, digest

RIDGE = 0.1  # Fixed before fitting; penalty on slope in log-time coordinates.
SCENARIO = 'unseen_column_and_molecules'


def correction(scales, residuals, test_scale):
    x=np.log(np.asarray(scales,dtype=float));y=np.asarray(residuals,dtype=float)
    if len(x)<2 or len(set(x))<2:raise ValueError('Two distinct calibration scales required')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(test_scale) or test_scale<=0:
        raise ValueError('Invalid calibration data')
    centered=x-x.mean()
    slope=float(np.dot(centered,y-y.mean())/(np.dot(centered,centered)+RIDGE))
    return float(y.mean()+slope*(np.log(test_scale)-x.mean())),slope


def groups(systems):
    graph=defaultdict(set)
    for a,b in method_pairs(systems):graph[a].add(b);graph[b].add(a)
    seen=set();result=[]
    for start in sorted(graph):
        if start in seen:continue
        pending=[start];component=set()
        while pending:
            sid=pending.pop()
            if sid in component:continue
            component.add(sid);pending.extend(graph[sid]-component)
        seen.update(component)
        if len(component)>=3:result.append(sorted(component))
    return result


def metrics(rows):
    if not rows:return {'n':0}
    return {'n':len(rows),**{name+'_mae_min':float(np.mean([abs(r[name]-r['observed_tr_min']) for r in rows]))
        for name in ('catboost','constant_offset','local_slope')},
        'local_clipped_to_zero':sum(r['local_unclipped']<0 for r in rows)}


def run():
    base=Path('reports/mcmrt_transfer_scaled');sp=Path('data/processed/public_retention/mcmrt/systems.json')
    systems=json.loads(sp.read_text());old_plan=json.loads((base/'plan.json').read_text())
    if hashlib.sha256(sp.read_bytes()).hexdigest()!=old_plan['systems_sha256']:raise ValueError('Metadata changed')
    rows=[json.loads(l) for l in (base/'predictions.jsonl').read_text().splitlines()]
    index={(r['system_id'],r['compound_id']):r for r in rows if r['scenario']==SCENARIO}
    folds={f['fold']:f for f in old_plan['splits'] if f['scenario']==SCENARIO}
    cases=[]
    for group in groups(systems):
        compounds=sorted(set.intersection(*[{c for sid,c in index if sid==s} for s in group]))
        for compound in compounds:
            for test in group:
                train=[s for s in group if s!=test]
                records=[index[(s,compound)] for s in group]
                if len({r['fold'] for r in records})!=1:raise ValueError('Different base models in one comparison')
                fold=folds[records[0]['fold']]
                for r in records:
                    if r['observation_id'] not in fold['test_ids'] or r['observation_id'] in fold['training_ids']:
                        raise ValueError('Column data leaked to base model')
                cases.append({'compound_id':compound,'test_system':test,'calibration_systems':train,
                              'test_id':index[(test,compound)]['observation_id'],
                              'calibration_ids':[index[(s,compound)]['observation_id'] for s in train]})
    plan={'prediction_sha256':hashlib.sha256((base/'predictions.jsonl').read_bytes()).hexdigest(),
          'base_plan_sha256':digest(old_plan),'ridge':RIDGE,'scenario':SCENARIO,'cases':cases,
          'model':'Per-molecule residual intercept and slope versus log(time to first maximum organic)',
          'baseline':'Per-molecule median residual; same calibration measurements',
          'scope':'Known locally measured molecules; unseen program. NOT unseen molecules after adaptation.',
          'selection':'All complete series with >=3 matching-condition programs; no outcome selection or tuning.'}
    out=Path('reports/local_gradient_model');out.mkdir(parents=True,exist_ok=True);pp=out/'plan.json'
    if pp.exists() and json.loads(pp.read_text())!=plan:raise ValueError('Frozen plan changed')
    if not pp.exists():pp.write_text(json.dumps(plan,indent=2)+'\n')
    predictions=[]
    for case in cases:
        molecule=case['compound_id'];test=index[(case['test_system'],molecule)]
        train=[index[(s,molecule)] for s in case['calibration_systems']]
        scales=[analytical_scale(systems[s]) for s in case['calibration_systems']]
        scale=analytical_scale(systems[case['test_system']])
        residuals=[r['observed_tr_min']-r['predicted_tr_min'] for r in train]
        delta,slope=correction(scales,residuals,scale)
        raw=test['predicted_tr_min']+delta
        predictions.append({**case,'observed_tr_min':test['observed_tr_min'],
            'catboost':test['predicted_tr_min'],
            'constant_offset':max(0.,test['predicted_tr_min']+float(np.median(residuals))),
            'local_slope':max(0.,raw),'local_unclipped':raw,'residual_slope':slope,
            'scale_mode':'interpolation' if min(scales)<=scale<=max(scales) else 'extrapolation',
            'calibration_program_count':len(train)})
    result={'overall':metrics(predictions),
            'by_test_program':{s:metrics([r for r in predictions if r['test_system']==s]) for s in sorted({r['test_system'] for r in predictions})},
            'by_scale_mode':{m:metrics([r for r in predictions if r['scale_mode']==m]) for m in ('interpolation','extrapolation')},
            'unique_molecules':len({r['compound_id'] for r in predictions}),
            'column_series':groups(systems),'production_approved':False,
            'limitations':['Retrospective development folds, not a new independent validation.',
              'Each analyte must have measured RT at 2 or 3 calibration programs on this column.',
              'No measurements from the held-out program enter fitting. Same molecules across calibration and test are intentional.',
              'Program changes are compressed to one analytical time scale; not a general gradient simulator.',
              'Interpolation refers only to time scale, not the full multivariate gradient shape.',
              'Same molecules recur across folds; rows are not independent trials.',
              'No TFV/FTC measurements, widths, Rs or calibrated uncertainty. Customer method unchanged.']}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    lines=['# Локальная модель изменения градиента','',
       'Для каждого вещества и колонки использованы 2–3 измеренные программы. Отдельная программа исключена из настройки. Базовый CatBoost не обучался на этой колонке и этих молекулах.',
       'После локальной настройки молекулы уже известны по измерениям: проверяется новая программа, а не новое вещество.',
       'Поправка: линейная зависимость ошибки CatBoost от логарифма времени первого максимума органики; коэффициент регуляризации 0,1 зафиксирован до расчёта.', '',
       '| Проверка | n | CatBoost MAE мин | Постоянная поправка | Локальная зависимость |','|---|---:|---:|---:|---:|']
    for name,m in [('Все',result['overall']),*result['by_scale_mode'].items(),*result['by_test_program'].items()]:
        if m['n']:lines.append(f"| {name} | {m['n']} | {m['catboost_mae_min']:.3f} | {m['constant_offset_mae_min']:.3f} | {m['local_slope_mae_min']:.3f} |")
    lines+=['',
            f"Отрицательные прогнозы локальной модели обрезаны до нуля: {result['overall']['local_clipped_to_zero']}. Это сбои переноса, а не достоверное отсутствие удерживания.",
            'Вывод текущего опыта: локальная зависимость улучшает исходный CatBoost в среднем, но уступает постоянной поправке. В рабочий предиктор не включена.',
            'Все варианты сравниваются на одних строках; улучшение оценивается относительно обеих базовых моделей.',
            'Для TFV/FTC такой поправки пока нет: отсутствуют локальные измерения выбранного метода.',*result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
