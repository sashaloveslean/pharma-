"""Paired gradient-response audit using frozen, same-fold held-out predictions."""
from collections import defaultdict
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_mcmrt_transfer import digest, analytical_scale

SCENARIOS = ('unseen_column_known_molecules', 'unseen_column_and_molecules')


def method_pairs(systems):
    groups = defaultdict(list)
    for sid, system in systems.items():
        if not system['eligible_for_rt_research']: continue
        flows = {p['flow rate [ml/min]'] for p in system['gradient_program']}
        if len(flows) != 1: continue  # Exclude confounding variable-flow programs.
        context = {'metadata': system['metadata'], 'instrument': system.get('instrument'),
                   'flow': next(iter(flows))}
        groups[digest(context)].append(sid)
    return [(a, b) for ids in groups.values() for a, b in combinations(sorted(ids), 2)
            if systems[a]['gradient_program'] != systems[b]['gradient_program']]


def summarize(rows):
    if not rows: return {'n': 0}
    meaningful = [r for r in rows if abs(r['observed_change_min']) > .1]
    return {'n': len(rows),
            'delta_mae_min': float(np.mean([abs(r['predicted_change_min']-r['observed_change_min']) for r in rows])),
            'no_change_mae_min': float(np.mean([abs(r['observed_change_min']) for r in rows])),
            'time_scaling_mae_min': float(np.mean([abs(r['time_scaling_change_min']-r['observed_change_min']) for r in rows])),
            'direction_n': len(meaningful),
            'time_scaling_direction_accuracy': float(np.mean([r['time_scaling_change_min']*r['observed_change_min'] > 0 for r in meaningful])) if meaningful else None,
            'direction_accuracy': float(np.mean([r['predicted_change_min']*r['observed_change_min'] > 0
                                                for r in meaningful])) if meaningful else None}


def run():
    root = Path('reports/mcmrt_transfer_scaled')
    system_path = Path('data/processed/public_retention/mcmrt/systems.json')
    systems = json.loads(system_path.read_text())
    old_plan = json.loads((root/'plan.json').read_text())
    if hashlib.sha256(system_path.read_bytes()).hexdigest() != old_plan['systems_sha256']:
        raise ValueError('Source systems changed')
    pairs = method_pairs(systems)
    rows = [json.loads(l) for l in (root/'predictions.jsonl').read_text().splitlines()]
    index = {(r['scenario'], r['system_id'], r['compound_id']): r for r in rows if r['scenario'] in SCENARIOS}
    if len(index) != sum(r['scenario'] in SCENARIOS for r in rows):
        raise ValueError('Ambiguous repeated structure/method')
    folds = {(f['scenario'], f['fold']): f for f in old_plan['splits']}
    out = Path('reports/gradient_change_audit'); out.mkdir(parents=True, exist_ok=True)
    plan = {'prediction_sha256': hashlib.sha256((root/'predictions.jsonl').read_bytes()).hexdigest(),
            'source_plan_sha256': digest(old_plan), 'pairs': pairs, 'scenarios': list(SCENARIOS),
            'direction_threshold_min': .1,
            'scope': 'Same metadata, instrument and constant flow; differing full gradient programs.',
            'note': 'Retrospective diagnostic on existing development folds; no new model tuning.'}
    p = out/'plan.json'
    # JSON round-trip normalizes tuple/list types.
    plan = json.loads(json.dumps(plan))
    if p.exists() and json.loads(p.read_text()) != plan: raise ValueError('Audit plan changed')
    if not p.exists(): p.write_text(json.dumps(plan, indent=2)+'\n')
    records = []
    for scenario in SCENARIOS:
        for a, b in pairs:
            compounds = sorted({key[2] for key in index if key[:2] == (scenario, a)} &
                               {key[2] for key in index if key[:2] == (scenario, b)})
            for compound in compounds:
                left, right = index[(scenario, a, compound)], index[(scenario, b, compound)]
                if left['fold'] != right['fold']: raise ValueError('Cannot compare different fitted models')
                fold = folds[(scenario, left['fold'])]
                for row in (left, right):
                    if row['observation_id'] not in fold['test_ids'] or row['observation_id'] in fold['training_ids']:
                        raise ValueError('Pair not fully held out')
                records.append({'scenario': scenario, 'from_system': a, 'to_system': b,
                    'compound_id': compound, 'fold': left['fold'],
                    'observed_change_min': right['observed_tr_min']-left['observed_tr_min'],
                    'predicted_change_min': right['predicted_tr_min']-left['predicted_tr_min'],
                    'time_scaling_change_min': left['observed_tr_min']*(analytical_scale(systems[b])/analytical_scale(systems[a])-1)})
    result = {'overall': {s: summarize([r for r in records if r['scenario']==s]) for s in SCENARIOS},
        'by_pair': {s: {a+' -> '+b: summarize([r for r in records if r['scenario']==s and r['from_system']==a and r['to_system']==b]) for a,b in pairs} for s in SCENARIOS},
        'small_observed_changes': {s: summarize([r for r in records if r['scenario']==s and .1 < abs(r['observed_change_min']) <= 2]) for s in SCENARIOS},
        'production_approved': False,
        'limitations': ['Repeated compounds/pairs are dependent observations, not independent trials.',
            'Both programs predicted by the same model with the entire column held out.',
            'Known-molecule scenario can use the same molecules on other columns; joint scenario excludes their connectivity globally.',
            'Time-scaling baseline uses one measured starting tR; it is an anchored baseline, not a zero-shot prediction.',
            'Matching recorded conditions cannot exclude unreported pH/dwell differences.',
            'TFV/FTC absent; changes to the customer gradient are not validated.',
            'Entire programs differ: this does not isolate one gradient segment or establish Rs.']}
    (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'paired_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    lines = ['# Проверка реакции модели на изменение градиента', '',
        'Сравниваются реальные измерения одного вещества при совпадающих колонке, фазах, температуре, постоянном потоке и указанном приборе.',
        'Для каждой пары оба прогноза получены одной моделью без обучения на этой колонке.', '',
        '| Сценарий | Пар наблюдений | Ошибка ΔtR, мин | Без изменения, мин | Масштабирование по времени, мин | Верное направление* |',
        '|---|---:|---:|---:|---:|---:|']
    for s, m in result['overall'].items():
        lines.append(f"| {s} | {m['n']} | {m['delta_mae_min']:.3f} | {m['no_change_mae_min']:.3f} | {m['time_scaling_mae_min']:.3f} | {m['direction_accuracy']:.1%} |")
    lines += ['', '*Для измеренного изменения более 0,1 мин; это порог анализа, не установленная экспериментальная погрешность.',
        *[f"{s}, небольшие измеренные изменения: MAE модели {m['delta_mae_min']:.3f} мин; предположение без изменения {m['no_change_mae_min']:.3f} мин; n={m['n']}." for s,m in result['small_observed_changes'].items()],
        'Вывод: для тонкой настройки метода преимущества пока не показано. Высокая общая доля верного направления не превосходит простое масштабирование.',
        'Ошибка ΔtR оценивает изменение удерживания одного вещества между программами, а не расстояние между пиками TFV и FTC.',
        'Масштабирование использует измеренный tR исходной программы. Пары и все ухудшения сохранены в evaluation.json.',
        'Для небольших измеренных изменений (0,1–2 мин) результаты приведены отдельно в JSON; это диагностическая подгруппа, не оценка точности малых правок градиента.',
        *[f"{s}: направление по простому масштабированию верно в {m['time_scaling_direction_accuracy']:.1%} случаев." for s,m in result['overall'].items()], '', *result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
