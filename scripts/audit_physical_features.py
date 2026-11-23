"""Evaluate the declared feature experiment gate without using external labels."""
import json
from pathlib import Path
from audit_mcmrt_errors import audit
from audit_early_exit import summarize
from evaluate_mcmrt_transfer import analytical_scale
from open_retention import load_observations


def gate(previous, current, config):
    joint = 'unseen_column_and_molecules'
    return bool(all(current[s]['all']['mae_min'] <= previous[s]['all']['mae_min'] *
                    config['max_overall_mae_ratio_each_scenario'] for s in previous)
                and previous[joint]['early']['n'] > 0
                and current[joint]['early']['n'] > 0
                and current[joint]['early']['mae_min'] < previous[joint]['early']['mae_min'])


def run():
    new = Path('reports/mcmrt_transfer_physical')
    plan = json.loads((new/'plan.json').read_text())
    config = plan['research_gate']; old = Path(config['reference'])
    prior_plan = json.loads((old/'plan.json').read_text())
    if plan['target_mode'] != prior_plan['target_mode']:
        raise ValueError('Feature experiment changed target')
    breakdown = audit(old, new)  # verifies cases, labels, data, folds and parameters
    data = load_observations(Path('data/processed/public_retention/mcmrt/observations.jsonl'))
    scales = {r['observation_id']: analytical_scale(r) for r in data}
    results = {}
    for name, folder in [('previous', old), ('candidate', new)]:
        rows = [json.loads(l) for l in (folder/'predictions.jsonl').read_text().splitlines()]
        results[name] = {}
        for scenario in sorted({r['scenario'] for r in rows}):
            part = [r for r in rows if r['scenario'] == scenario]
            early = [r for r in part if r['observed_tr_min']/scales[r['observation_id']] <= config['early_fraction']]
            results[name][scenario] = {'all': summarize(part), 'early': summarize(early)}
    passed = gate(results['previous'], results['candidate'], config)
    result = dict(results=results, breakdown=breakdown, passes_internal_research_gate=passed,
                  production_approved=False, external_labels_used_for_selection=False,
                  feature_version=plan['feature_version'], criteria=config)
    out = Path('reports/physical_features_audit'); out.mkdir(parents=True, exist_ok=True)
    (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Проверка физических признаков', '',
             'Данные, разбиения, целевая переменная и параметры модели совпадают. Изменены только признаки.', '',
             '| Сценарий / группа | Было MAE, мин | Стало MAE, мин | n |', '|---|---:|---:|---:|']
    for s, parts in results['previous'].items():
        for group, before in parts.items():
            after = results['candidate'][s][group]
            lines.append(f"| {s} / {group} | {before['mae_min']:.3f} | {after['mae_min']:.3f} | {after['n']} |")
    lines += ['', f'Предварительный критерий пройден: {passed}.',
              'Критерий: улучшение ранних пиков в совместном переносе; ухудшение общего MAE не более 5% в каждом сценарии.',
              'Ранние пики: tR не более 10% времени первого максимума органики; это не измерение t0.',
              'Объём колонки / поток — геометрический масштаб, не измеренное мёртвое время. Формальный заряд не заменяет профиль ионизации при pH.',
              'Эти группы уже использовались при разработке: это исследовательское сравнение, не независимая валидация.',
              'TFV и FTC в MCMRT отсутствуют. Рабочая модель и выбранный градиент не изменены.',
              'При провале критерия внешняя проверка кандидата не запускается.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='breakdown'}, indent=2))


if __name__ == '__main__': run()
