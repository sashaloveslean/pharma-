"""Audit local open retention metadata; coverage is not evidence of a causal pH effect."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

SALTS = ('nh4form', 'nh4ac', 'nh4carb', 'nh4bicarb')
TARGETS = {'TFV': 'SGOIRFVFHAKUTI-ZCFIWIBFSA-N', 'FTC': 'XQSPYNMVSIKCOC-NTSWFWBYSA-N'}


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def phase_record(metadata, channel, source):
    prefix = f'eluent.{channel}.'
    ph = number(metadata.get(prefix + 'pH'))
    ph = ph if ph is not None and 0 <= ph <= 14 else None
    salts = []
    for salt in SALTS:
        key = prefix + salt
        value = number(metadata.get(key))
        if value is None or value <= 0:
            continue
        unit = metadata.get(key + '.unit', '')
        # import_mcmrt.phase explicitly parses mM from the supplementary recipes.
        provenance = 'explicit_unit'
        if not unit and source == 'mcmrt':
            unit, provenance = 'mM', 'mcmrt_importer_parses_mM'
        normalized = value if unit == 'mM' else value * 1000 if unit == 'M' else None
        salts.append(dict(salt=salt, value=value, unit=unit, concentration_mM=normalized,
                          unit_provenance=provenance))
    return {'channel': channel, 'reported_pH': ph, 'water_percent': number(metadata.get(prefix+'h2o')), 'salts': salts}


def active_channels(system):
    points = system.get('gradient_program', [])
    return [c for c in 'ABCD' if any((number(p.get(f'{c} [%]')) or 0) > 0 for p in points)]


def signature_without_ph(system):
    """Strict metadata match, excluding IDs/t0/pH; incomplete matches still need review."""
    m = {k: number(v) if number(v) is not None else v
         for k, v in system['metadata'].items()
         if k not in ('id', 'column.t0') and not k.endswith('.pH')}
    return json.dumps([m, system.get('gradient_program'), system.get('instrument')], sort_keys=True)


def run(root, out):
    rows, summary, matched = [], {}, []
    for path in sorted(root.glob('*/systems.json')):
        source = path.parent.name
        systems = json.loads(path.read_text())
        counts, molecules = Counter(), defaultdict(set)
        row_ph_count = 0
        target_rows = defaultdict(Counter)
        observations = path.parent/'observations.jsonl'
        if observations.exists():
            with observations.open() as stream:
                for line in stream:
                    r = json.loads(line)
                    rt = number(r.get('retention_time_min'))
                    if rt is None or rt <= 0:
                        continue
                    if any(k.endswith('.pH') and number(v) is not None and 0 <= number(v) <= 14
                           for k, v in r.get('metadata', {}).items()):
                        row_ph_count += 1
                    sid = r['system_id']; counts[sid] += 1
                    molecules[sid].add(r.get('compound_id'))
                    for target, key in TARGETS.items():
                        if r.get('compound_id') == key:
                            target_rows[sid][target] += 1
        groups = defaultdict(list)
        source_rows = []
        for sid, system in systems.items():
            channels = active_channels(system)
            phases = [phase_record(system['metadata'], c, source) for c in channels]
            # Only a pure aqueous reservoir gives an unambiguous aqueous-pH label.
            aqueous = [p for p in phases if p['water_percent'] == 100]
            qualified = [p for p in aqueous if p['reported_pH'] is not None and p['salts']
                         and all(s['concentration_mM'] is not None for s in p['salts'])]
            info = system.get('source_info', [])
            record = dict(source=source, system_id=sid, rt_observations=counts[sid],
                unique_molecules=len(molecules[sid]), phases=phases,
                rp=any(i.get('method.type') == 'RP' for i in info),
                eligible=system.get('eligible_for_rt_research', True),
                reported_ph=any(p['reported_pH'] is not None for p in phases),
                salt_present=any(p['salts'] for p in phases),
                aqueous_ph_and_salt_units=bool(qualified),
                source_urls=sorted({i['url'] for i in info if i.get('url')}),
                exact_target_observations=dict(target_rows[sid]))
            rows.append(record); source_rows.append(record)
            if record['rp'] and record['eligible'] and counts[sid] and qualified:
                groups[signature_without_ph(system)].append(sid)
        for ids in groups.values():
            phs = {tuple((c, phase_record(systems[s]['metadata'], c, source)['reported_pH'])
                         for c in active_channels(systems[s])) for s in ids}
            if len(ids) > 1 and len(phs) > 1:
                shared = set.intersection(*(molecules[s] for s in ids))
                matched.append(dict(source=source, systems=ids, shared_molecules=len(shared),
                    status='metadata_matched_only_not_causal_proof'))
        rp = [r for r in source_rows if r['rp'] and r['eligible'] and r['rt_observations']]
        complete = [r for r in rp if r['aqueous_ph_and_salt_units']]
        summary[source] = dict(observations_with_row_level_ph=row_ph_count, total_systems=len(systems), rp_eligible_with_rt=len(rp),
            rp_reported_ph=sum(r['reported_ph'] for r in rp),
            rp_salt_present=sum(r['salt_present'] for r in rp),
            rp_aqueous_ph_and_salt_units=len(complete),
            observations_in_these_systems=sum(r['rt_observations'] for r in complete),
            source_urls_in_these_systems=len({u for r in complete for u in r['source_urls']}),
            exact_targets_in_these_systems={t:sum(r['exact_target_observations'].get(t,0) for r in complete) for t in TARGETS})
    result = dict(sources=summary, strict_ph_comparison_groups=matched,
        model_audit={'model':'evaluate_mcmrt_transfer.feature/base',
                     'salt_concentrations_already_features':True, 'pH_is_feature':False,
                     'buffer_effect_validated':False},
        decision='Curate aqueous pH + salt subset and validate source-disjoint models before phase ranking.',
        limitations=['Reported pH is not necessarily measured pH; no pH inferred from acid concentration.',
                     'System-level counts do not include variable row-level metadata; inspect observations_with_row_level_ph.',
                     'Positive salt concentration does not prove buffering capacity.',
                     'Counts are raw observations, not independent experiments or deduplicated training rows.',
                     'Matched metadata can omit instrument details and does not establish causation.',
                     'No new model trained; no peak shape or Rs prediction.'])
    out.mkdir(parents=True, exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'systems.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    lines=['# Аудит pH и буферов в локальных открытых данных', '',
           'Единица подсчёта — система/метод, не независимый эксперимент. Поднабор ниже требует RP, положительного tR, водного резервуара 100% H2O, указанного pH и положительной концентрации соли с известными единицами.', '',
           '| Набор | RP-методы с tR | С pH | С солью | Водный pH + соль с единицами | Строк tR в поднаборе |',
           '|---|---:|---:|---:|---:|---:|']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['rp_eligible_with_rt']} | {s['rp_reported_ph']} | {s['rp_salt_present']} | {s['rp_aqueous_ph_and_salt_units']} | {s['observations_in_these_systems']} |")
    lines += ['', 'Важно: таблица выше учитывает метаданные систем. Построчный pH найден отдельно: ' + str({k:v['observations_with_row_level_ph'] for k,v in summary.items()}) + '. Ноль в таблице не означает отсутствия pH во всём наборе. Для U Liège см. reports/uliege_ph_recovery/REPORT.md.', '', f'Групп с одинаковыми остальными записанными условиями и разными pH: {len(matched)}. Это строгий поиск по доступным метаданным; нулевой результат не доказывает отсутствие таких опытов в исходных публикациях.', '',
              'В текущей MCMRT-модели концентрации формиата и ацетата уже являются признаками. pH признаком не является. Наличие признака не означает доказанную способность оптимизировать его.', '',
              'Следующий этап: проверить первоисточники найденного поднабора, единицы и происхождение pH; убрать повторы; сравнить одинаковые модели с признаками pH/солей и без них на разбиении по публикациям. До успешной проверки не выдавать модельный рейтинг буферов.', '',
              'В audit.json сохранены точные совпадения TFV/FTC, показатели покрытия и ограничения. systems.jsonl содержит составы, ссылки и идентификаторы для проверки. Никакого дообучения и новых прогнозов на этом этапе нет.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/processed/public_retention'))
    p.add_argument('--out', type=Path, default=Path('reports/buffer_ph_audit'))
    a=p.parse_args(); print(json.dumps(run(a.root,a.out),ensure_ascii=False,indent=2))
