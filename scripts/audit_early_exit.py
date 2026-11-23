"""Predeclared MCMRT-only gate; external labels never select this candidate."""
import json
from pathlib import Path
import numpy as np
from open_retention import load_observations
from evaluate_mcmrt_transfer import analytical_scale
from audit_mcmrt_errors import audit


def summarize(rows):
    if not rows:return {'n':0,'mae_min':None,'bias_min':None}
    return {'n':len(rows),'mae_min':float(np.mean([r['absolute_error_min'] for r in rows])),
            'bias_min':float(np.mean([r['predicted_tr_min']-r['observed_tr_min'] for r in rows]))}


def gate(previous,current):
    if not previous['early']['n'] or not current['early']['n']:return False
    return (current['early']['mae_min']<previous['early']['mae_min'] and
            current['all']['mae_min']<=previous['all']['mae_min']*1.05)


def run():
    old=Path('reports/mcmrt_transfer_scaled');new=Path('reports/mcmrt_transfer_log')
    audit(old,new) # Reject changes to data, folds, hyperparameters or labels.
    data=load_observations(Path('data/processed/public_retention/mcmrt/observations.jsonl'))
    scales={r['observation_id']:analytical_scale(r) for r in data}
    results={}
    for name,folder in [('previous',old),('candidate',new)]:
        rows=[json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        results[name]={}
        for scenario in sorted({r['scenario'] for r in rows}):
            part=[r for r in rows if r['scenario']==scenario]
            early=[r for r in part if r['observed_tr_min']/scales[r['observation_id']]<=.1]
            other=[r for r in part if r['observed_tr_min']/scales[r['observation_id']]>.1]
            results[name][scenario]={'all':summarize(part),'early':summarize(early),'other':summarize(other)}
    scenario='unseen_column_and_molecules'
    result={'results':results,'passes_internal_research_gate':gate(results['previous'][scenario],results['candidate'][scenario]),
            'production_approved':False,'external_labels_used_for_selection':False,
            'limitations':['Early = first 10% of analytical scale, not measured void time or proven weak retention.',
                'MCMRT groups already informed development. Gate is a research screen, not independent validation.',
                'No external FTC prediction is used to choose the change.']}
    out=Path('reports/early_exit_audit');out.mkdir(parents=True,exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Ранний выход: проверка изменения целевой переменной','',
          '| Сценарий / часть | Было MAE, мин | Стало MAE, мин | Число |','|---|---:|---:|---:|']
    for s,parts in results['previous'].items():
        for name,previous in parts.items():
            current=results['candidate'][s][name]
            text.append(f"| {s} / {name} | {previous['mae_min']:.3f} | {current['mae_min']:.3f} | {current['n']} |")
    text+=['',f"Предварительный критерий MCMRT пройден: {result['passes_internal_research_gate']}.",
           'Критерий: улучшение ранних пиков для новых колонок и молекул при ухудшении общего MAE не более 5%. Он записан в плане до обучения.',*result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
