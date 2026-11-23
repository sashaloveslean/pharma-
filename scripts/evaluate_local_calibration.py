"""Fixed 20-compound median residual calibration; retrospective research only."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluate_external_transfer import adapt_system
from evaluate_mcmrt_transfer import digest

TARGETS = {'SGOIRFVFHAKUTI', 'XQSPYNMVSIKCOC'}
N_CALIBRATION = 20


def connectivity(row):
    return row['compound_id'].split('-')[0]


def partition(rows, method_ids):
    pools = defaultdict(set)
    for r in rows:
        if connectivity(r) not in TARGETS:
            pools[method_ids[r['system_id']]].add(connectivity(r))
    selected = {}
    for method, molecules in sorted(pools.items()):
        if len(molecules) <= N_CALIBRATION:
            raise ValueError('Insufficient independent molecules')
        selected[method] = sorted(molecules, key=lambda m: hashlib.sha256(
            ('local-calibration-v1:' + m).encode()).hexdigest())[:N_CALIBRATION]
    all_calibration = set().union(*map(set, selected.values()))
    fit = [r for r in rows if connectivity(r) in selected[method_ids[r['system_id']]]]
    test = [r for r in rows if connectivity(r) not in all_calibration]
    return selected, fit, test


def fit_offsets(rows, method_ids):
    # Equal molecule weight: multiple observations/stereoisomers do not get extra votes.
    residuals = defaultdict(list)
    for r in rows:
        residuals[(method_ids[r['system_id']], connectivity(r))].append(
            r['observed_tr_min'] - r['scaled_tr_min'])
    grouped = defaultdict(list)
    for (method, molecule), values in residuals.items():
        grouped[method].append(float(np.median(values)))
    return {method: float(np.median(values)) for method, values in grouped.items()}


def metrics(rows):
    if not rows: return {'n': 0}
    return {'n': len(rows), **{name+'_mae_min': float(np.mean([
        abs(r[field] - r['observed_tr_min']) for r in rows])) for name, field in
        [('before', 'scaled_tr_min'), ('after', 'calibrated_tr_min')]},
        'clipped_to_zero': sum(r['unclipped_tr_min'] < 0 for r in rows)}


def run():
    source = Path('reports/external_transfer/predictions.jsonl')
    system_path = Path('data/processed/public_retention/report/systems.json')
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    systems = json.loads(system_path.read_text())
    method_ids = {}
    for sid in sorted({r['system_id'] for r in rows}):
        system = adapt_system(systems[sid])
        method_ids[sid] = digest({'metadata': system['metadata'], 'program': system['gradient_program']})
    selected, fit, test = partition(rows, method_ids)
    plan = {'prediction_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'systems_sha256': hashlib.sha256(system_path.read_bytes()).hexdigest(),
            'method_ids': method_ids, 'calibration_molecules': selected,
            'fit_ids': [r['observation_id'] for r in fit],
            'test_ids': [r['observation_id'] for r in test],
            'estimator': 'median of per-connectivity median residuals, additive minutes; clip output at zero',
            'n_calibration_per_method': N_CALIBRATION,
            'selection': 'fixed structure hash, no retention-time or error selection',
            'excluded_targets': sorted(TARGETS), 'production_approved': False}
    out = Path('reports/local_calibration'); out.mkdir(parents=True, exist_ok=True)
    plan_path = out/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen calibration plan changed')
    if not plan_path.exists(): plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    offsets = fit_offsets(fit, method_ids)
    predictions = []
    for r in test:
        raw = r['scaled_tr_min'] + offsets[method_ids[r['system_id']]]
        predictions.append({**r, 'unclipped_tr_min': raw, 'calibrated_tr_min': max(0., raw)})
    result = {'overall': metrics(predictions),
              'by_system': {sid: metrics([r for r in predictions if r['system_id'] == sid]) for sid in method_ids},
              'offsets_min': offsets, 'calibration_rows': len(fit),
              'excluded_from_test_rows': len(rows)-len(test),
              'target_diagnostics': [r for r in predictions if connectivity(r) in TARGETS],
              'production_approved': False,
              'limitations': ['Retrospective adaptation on previously inspected public sources; not a fresh blind test.',
                 'Same complete method only. Does not calibrate arbitrary gradients, columns or the customer instrument.',
                 'No TFV/FTC labels enter calibration; target results are diagnostics, not selection criteria.',
                 'One fixed structure-based calibration panel; sensitivity to panel choice remains untested.',
                 'No measured t0, peak widths, Rs or validated prediction intervals.',
                 'Zero-clipped predictions are boundary failures, not evidence of correct retention.']}
    (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
    lines = ['# Локальная калибровка по открытым измерениям', '',
             '20 веществ на полный метод. Поправка — медиана ошибок; выбор веществ по хешу структуры, без использования времени удерживания.',
             'Все калибровочные структуры исключены из проверки на всех методах. Базовая модель не переобучалась.', '',
             '| Группа | n | MAE до, мин | MAE после, мин | Прогнозов на границе 0 |', '|---|---:|---:|---:|---:|']
    for name, m in [('Все', result['overall']), *result['by_system'].items()]:
        lines.append(f"| {name} | {m['n']} | {m['before_mae_min']:.3f} | {m['after_mae_min']:.3f} | {m['clipped_to_zero']} |")
    lines += ['', 'Целевые вещества (отдельные условия источника, не метод заказчика):']
    for r in result['target_diagnostics']:
        lines.append(f"{r['compound_id']}, {r['system_id']}: измерено {r['observed_tr_min']:.4f}; до {r['scaled_tr_min']:.4f}; после {r['calibrated_tr_min']:.4f} мин.")
    if not any(connectivity(r) == 'SGOIRFVFHAKUTI' for r in predictions):
        lines.append('TFV: измерений для проверки нет.')
    lines += ['', *result['limitations'], '', 'Рабочая модель и выбранный градиент TFV/FTC не изменены.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
