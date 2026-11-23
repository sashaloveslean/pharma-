"""Evidence and deterministic sensitivity diagnostics, not calibrated probabilities."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from evaluate_mcmrt_transfer import feature, analytical_scale
from evaluate_external_transfer import adapt_system
from propose_pair_method import check_method

SCENARIOS=[('middle_B',-2),('middle_B',2),('knee_min',-.05),('knee_min',.05),
           ('hold_min',-.1),('hold_min',.1),('end_min',-.05),('end_min',.05)]


def perturb(parameters,key,change):
    p=deepcopy(parameters)
    p[key]=p[key]*(1+change) if key in ('knee_min','end_min') else p[key]+change
    return p


def confidence_level(column_seen,local_validation,rank_stable):
    # Even a stable numerical prediction does not demonstrate empirical accuracy.
    if not local_validation:return 'low'
    return 'medium' if column_seen and rank_stable else 'low'


def support(system, training):
    eligible=[s for s in training.values() if s.get('eligible_for_rt_research')]
    m=system['metadata']
    same=[s for s in eligible if s['metadata']['column.name']==m['column.name']]
    phase_keys=[f'eluent.{ch}.{k}' for ch in 'AB' for k in ('h2o','meoh','acn','formic','nh4form','nh4ac')]
    phase_match=[s for s in eligible if all(float(s['metadata'][k])==float(m[k]) for k in phase_keys)]
    return {'eligible_training_methods':len(eligible),'same_column_label_methods':len(same),
            'same_phase_recipe_methods':len(phase_match),
            'column_match_policy':'Exact source label; no inferred chemical equivalence',
            'local_empirical_validation':False}


def attach(rows,system,model,structures,make_program,model_training_hash):
    path=Path('data/processed/public_retention/mcmrt/systems.json')
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model_training_hash:
        raise ValueError('Training metadata differs from frozen model plan')
    domain=support(system,json.loads(path.read_text()))
    records=[]; owners=[]
    for i,row in enumerate(rows):
        for j,(key,change) in enumerate(SCENARIOS):
            p=perturb(row['parameters'],key,change)
            gradient=make_program(p['start_B'],p['hold_min'],p['middle_B'],p['knee_min'],p['end_min'],float(system['metadata']['column.flowrate']))
            if check_method(system['metadata'],gradient):raise ValueError('Invalid sensitivity scenario')
            records.append(adapt_system({**system,'gradient_program':gradient}));owners.append((i,j))
    scales=np.array([analytical_scale(r) for r in records])
    predictions={name:np.maximum(0,np.expm1(model.predict([feature({**r,'smiles':smiles}) for r in records])))*scales for name,smiles in structures.items()}
    scores=np.full((len(rows),len(SCENARIOS)),np.nan)
    gaps=np.zeros_like(scores); times={name:np.zeros_like(scores) for name in structures}
    valid=np.zeros_like(scores,dtype=bool)
    for k,(i,j) in enumerate(owners):
        vals=[predictions[name][k] for name in structures]
        for name in structures:times[name][i,j]=predictions[name][k]
        gaps[i,j]=abs(vals[0]-vals[1])
        valid[i,j]=all(np.isfinite(v) and 0<v<scales[k] for v in vals)
        if valid[i,j]:scores[i,j]=gaps[i,j]/scales[k]
    # Strict numerical ranks, with candidate-ID tie breaking; not probabilities.
    ranks=np.full_like(scores,np.nan)
    for j in range(len(SCENARIOS)):
        order=sorted([i for i in range(len(rows)) if valid[i,j]],key=lambda i:(-scores[i,j],rows[i]['candidate_id']))
        for rank,i in enumerate(order,1):ranks[i,j]=rank
    for i,row in enumerate(rows):
        rr=ranks[i,np.isfinite(ranks[i])]
        stable=len(rr)==len(SCENARIOS) and max(rr)-min(rr)<=5
        reasons=['Нет независимой проверки TFV/FTC на этих программах.',
                 'Нет калиброванных интервалов или модели ширин пиков; вероятность разделения не определена.']
        if not domain['same_column_label_methods']:reasons.append('Точного названия колонки нет среди обучающих методов MCMRT.')
        if not stable:reasons.append('Численный ранг меняется при малых изменениях градиента.')
        row['confidence']={'level':confidence_level(bool(domain['same_column_label_methods']),False,stable),
            'kind':'heuristic_evidence_assessment','probability':None,'prediction_interval':None,
            'reasons':reasons,'domain':domain,
            'sensitivity':{'scenario_count':len(SCENARIOS),'valid_scenarios':int(valid[i].sum()),
                'rank_scope':'All generated candidates; score = predicted gap / analytical duration',
                'rank_min':int(min(rr)) if len(rr) else None,'rank_max':int(max(rr)) if len(rr) else None,
                'rank_stable_heuristic':bool(stable),
                'predicted_gap_range_min':[float(gaps[i].min()),float(gaps[i].max())],
                'predicted_tr_ranges_min':{name:[float(values[i].min()),float(values[i].max())] for name,values in times.items()},
                'range_meaning':'Deterministic input perturbations, NOT an uncertainty/confidence interval'}}
    return {'version':'evidence_sensitivity_v1','training_systems_sha256':model_training_hash,
            'scenarios':SCENARIOS,'rank_stability_rule':'All 8 scenarios valid and rank range <= 5; heuristic, not calibrated',
            'domain':domain,'probabilistic_confidence_available':False}
