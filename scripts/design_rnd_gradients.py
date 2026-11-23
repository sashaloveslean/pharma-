"""Generate new TFV/FTC gradient hypotheses; research screening, never proven Rs."""
import argparse
from copy import deepcopy
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem
from evaluate_external_transfer import adapt_system
from evaluate_mcmrt_transfer import feature, analytical_scale, digest
from propose_pair_method import check_method

TARGETS={'tenofovir':'SGOIRFVFHAKUTI-ZCFIWIBFSA-N','emtricitabine':'XQSPYNMVSIKCOC-NTSWFWBYSA-N'}


def program(start, hold, middle, knee, end, flow=.2):
    if not (0 < hold < knee < end and 10 <= start < middle < 95):
        raise ValueError('Invalid gradient design')
    # Preserve source durations of wash, return and re-equilibration, not their adequacy.
    return [{'t [min]':round(t,4),'A [%]':100-b,'B [%]':b,'C [%]':0.,'D [%]':0.,
             'flow rate [ml/min]':flow} for t,b in
            [(0,start),(hold,start),(knee,middle),(end,95),(end+8,95),(end+8.1,start),(end+13,start)]]


def generate(system):
    results=[]
    for start,hold,middle,knee,end in product((10,15,20),(.5,1.,2.),(35,50,65),(4.,6.,8.),(12.,17.)):
        p=program(start,hold,middle,knee,end,float(system['metadata']['column.flowrate']))
        issues=check_method(system['metadata'],p)
        if issues: raise ValueError('Invalid generated conditions: '+str(issues))
        results.append({'candidate_id':'rnd:'+digest(p)[:12],'origin':'generated_hypothesis',
                        'parameters':dict(start_B=start,hold_min=hold,middle_B=middle,knee_min=knee,end_min=end),
                        'gradient_program':p})
    return results


def assess(predictions,end):
    finite=all(np.isfinite(v) and v>0 for v in predictions.values())
    inside=finite and all(v<end for v in predictions.values())
    gap=abs(predictions['tenofovir']-predictions['emtricitabine']) if finite else None
    return {'predicted_gap_min':gap,'eligible_for_heuristic_ranking':inside,
            'gap_per_analytical_min':gap/end if inside else None,
            'exclusion_reason':None if inside else 'nonpositive_nonfinite_or_peak_after_analytical_ramp'}


def choose(rows):
    eligible=[r for r in rows if r['eligible_for_heuristic_ranking']]
    if not eligible:return []
    # Fixed exploratory roles, not three claims of an optimum.
    selected=[]
    for role,key in [('gap_per_time',lambda r:(-r['gap_per_analytical_min'],r['candidate_id'])),
                     ('short_program',lambda r:(r['parameters']['end_min'],-r['gap_per_analytical_min'],r['candidate_id'])),
                     ('largest_predicted_gap',lambda r:(-r['predicted_gap_min'],r['candidate_id']))]:
        remaining=[r for r in eligible if r['candidate_id'] not in {x['candidate_id'] for x in selected}
                   and all(sum(r['parameters'][k]!=x['parameters'][k] for k in r['parameters'])>=2 for x in selected)]
        if remaining:selected.append({**min(remaining,key=key),'selection_role':role})
    return selected


def run(out):
    source=Path('data/processed/public_retention/report/systems.json')
    systems=json.loads(source.read_text());system=systems['report:0386']
    generated=generate(system)
    def signature(points):
        return digest([{k:float(p[k]) for k in ('t [min]','A [%]','B [%]','C [%]','D [%]','flow rate [ml/min]')} for p in points])
    known=set(); uncomparable=0
    for source_system in systems.values():
        try: known.add(signature(source_system['gradient_program']))
        except (ValueError, KeyError, TypeError): uncomparable+=1
    # Exact-program novelty only against parseable stored programs, not worldwide novelty.
    generated=[r for r in generated if signature(r['gradient_program']) not in known]
    model_path=Path('reports/external_transfer/research_scaled.cbm')
    out.mkdir(parents=True,exist_ok=True)
    plan={'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
          'systems_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'fixed_system':'report:0386','targets':TARGETS,'generated_count':len(generated),'source_programs_uncomparable':uncomparable,
          'search_space':{'start_B':[10,15,20],'hold_min':[.5,1,2],'middle_B':[35,50,65],'knee_min':[4,6,8],'end_min':[12,17]},
          'selection':'Exploratory roles: gap/analytical-time, short program, largest gap; at least two design parameters differ.',
          'constraints':'Same column, phase recipes, flow, temperature; existing column checks; wash 8 min, return .1 min, equilibration 4.9 min.',
          'uncertainty':'Unknown for generated programs; no calibrated interval or claimed Rs.',
          'production_approved':False}
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Design plan changed; choose a new output directory')
    if not path.exists():path.write_text(json.dumps(plan,indent=2)+'\n')
    structures={}
    for name,key in TARGETS.items():
        prop=json.loads(Path(f'reports/tenofovir_demo/pubchem_{name}.json').read_text())['payload']['PropertyTable']['Properties'][0]
        if prop['InChIKey']!=key or Chem.MolToInchiKey(Chem.MolFromSmiles(prop['SMILES']))!=key:raise ValueError('Wrong target identity')
        structures[name]=prop['SMILES']
    model=CatBoostRegressor();model.load_model(str(model_path))
    baseline={'candidate_id':'source:0386','origin':'source_control','gradient_program':system['gradient_program']}
    rows=[baseline]+generated; adapted=[]
    for r in rows:
        s=adapt_system({**system,'gradient_program':r['gradient_program']});adapted.append(s)
    for name,smiles in structures.items():
        X=[feature({**s,'smiles':smiles}) for s in adapted]
        predicted=np.maximum(0,np.expm1(model.predict(X)))*np.array([analytical_scale(s) for s in adapted])
        for r,value in zip(rows,predicted):r.setdefault('predicted_tr_min',{})[name]=float(value)
    for r,s in zip(rows,adapted):
        r.update(assess(r['predicted_tr_min'],analytical_scale(s)))
        r['predicted_Rs']=None;r['uncertainty']='unvalidated';r['separation_confirmed']=False
        r['cycle_min']=float(r['gradient_program'][-1]['t [min]'])
        r['solvent_volume_ml']=r['cycle_min']*.2
        r['local_calibration_applied']=False
        r['label_origin']='model_prediction';r['training_approved']=False
    from rnd_confidence import attach
    training_plan=json.loads(Path('reports/external_transfer/plan.json').read_text())
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=plan['model_sha256']:
        raise ValueError('Model changed during run')
    confidence_protocol=attach(generated,system,model,structures,program,training_plan['train_systems_sha256'])
    selected=choose(generated)
    for r in selected:
        r['change_vs_control']={'predicted_gap_min':r['predicted_gap_min']-baseline['predicted_gap_min'],
                                'cycle_min':r['cycle_min']-baseline['cycle_min']}
        r['demonstrated_improvement']=False
    result={'status':'rnd_hypotheses_for_experiments_not_optimized_method','confidence_protocol':confidence_protocol,'fixed_conditions':system['metadata'],
            'baseline':baseline,'generated_count':len(generated),'source_programs_uncomparable':uncomparable,'rankable_count':sum(r['eligible_for_heuristic_ranking'] for r in generated),
            'selected':selected,'instrument_execution_authorized':False,
            'limitations':['Prediction gap is not resolution; peak widths and measured t0 missing.',
              'Ranking can exploit prediction errors. Fine gradient tuning previously failed a meaningful baseline.',
              'Source pH convention and instrument dwell volume require verification; pH not a learned feature here.',
              'No calibrated uncertainty or validated pressure prediction; geometry checks are not instrument approval.',
              'Wash/return durations preserved as a design assumption; adequacy at new initial composition untested.',
              'Source and generated programs are separate. Novelty only checked against exact stored RepoRT programs.',
              'No target observations or synthetic training labels used; model and production recipe unchanged.']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generated))
    for i,r in enumerate([baseline]+selected):
        with (out/f'gradient_{i}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(r['gradient_program'][0]));w.writeheader();w.writerows(r['gradient_program'])
    lines=['# Собственные градиенты для R&D TFV и FTC','',
      f'Сгенерировано {len(generated)} новых программ; совпадения с численно читаемыми программами в локальном RepoRT исключены. Нечитаемых программ: {uncomparable}.',
      'Фиксированы Acclaim RSLC 120 C18 100 × 2,1 мм, 2,2 мкм; 30 °C; 0,20 мл/мин; A вода + 0,1% муравьиной кислоты, B метанол + 0,1% муравьиной кислоты. pH 2,6 по источнику, не измерение в нашей системе.',
      'Ищем начальный %B, выдержку, промежуточный %B и время достижения 95% B. Это собственные программы, а не равномерное растяжение исходной.',
      'Различия оценок очень малы: лучшая по разнице tR программа практически совпадает с контролем. Выбор не означает статистически значимого улучшения. Два коротких варианта проверяют гипотезу экономии времени при приемлемом разделении.',
      'Три роли — исследовательские гипотезы, не ранжирование по доказанной вероятности разделения.','',
      '| Вариант | TFV мин | FTC мин | Разница мин | Цикл мин |','|---|---:|---:|---:|---:|']
    for i,r in enumerate([baseline]+selected):
        lines.append(f"| {i}: {r.get('selection_role','исходный контроль')} | {r['predicted_tr_min']['tenofovir']:.2f} | {r['predicted_tr_min']['emtricitabine']:.2f} | {r['predicted_gap_min']:.2f} | {r['cycle_min']:.1f} |")
    for i,r in enumerate(selected,1):
        lines += ['',f'## Гипотеза {i}',
            f"Надёжность: {r['confidence']['level']} (качественная оценка, не вероятность). Ранг по gap/time при 8 возмущениях: {r['confidence']['sensitivity']['rank_min']}–{r['confidence']['sensitivity']['rank_max']} среди {len(generated)} программ.",
            *r['confidence']['reasons'],
            f"Диапазон прогнозируемой разницы tR при возмущениях: {r['confidence']['sensitivity']['predicted_gap_range_min']}. Это чувствительность к условиям, не доверительный интервал.",
            f"Параметры: {json.dumps(r['parameters'],ensure_ascii=False)}",'','| Время мин | B % |','|---|---:|']
        lines += [f"| {p['t [min]']} | {p['B [%]']} |" for p in r['gradient_program']]
    lines += ['', '## Как проверять',
      'Сначала исходный контроль и отдельные стандарты, затем выбранные гипотезы с повторными инъекциями. Записать реальные tR, ширины, Rs, давление и перенос. Исходный контроль не заменён.',
      'Увеличение предсказанного расстояния между пиками не доказывает улучшение. Неопределённость для новых программ не откалибрована. Эти гипотезы не становятся обучающими измерениями.',*result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'generated':len(generated),'baseline':baseline['predicted_tr_min'],'selected':[{k:r[k] for k in ('candidate_id','parameters','selection_role','predicted_tr_min','predicted_gap_min')} for r in selected]},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('reports/rnd_gradient_design'))
    run(parser.parse_args().out)
