"""Paired error breakdowns on identical frozen transfer cases; no model selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def audit(baseline,candidate=None):
    old=[json.loads(l) for l in (baseline/'predictions.jsonl').read_text().splitlines()]
    new=[json.loads(l) for l in (candidate/'predictions.jsonl').read_text().splitlines()] if candidate else old
    key=lambda r:(r['scenario'],r['fold'],r['observation_id'])
    index={key(r):r for r in new}
    if len(index)!=len(new) or {key(r) for r in old}!=set(index):raise ValueError('Comparison cases differ')
    if candidate:
        a=json.loads((baseline/'plan.json').read_text());b=json.loads((candidate/'plan.json').read_text())
        for field in ('data_sha256','systems_sha256','splits','params','feature_grid_min'):
            if a[field]!=b[field]:raise ValueError('Comparison changed '+field)
    groups=defaultdict(list)
    for row in old:
        other=index[key(row)]
        if row['observed_tr_min']!=other['observed_tr_min']:raise ValueError('Observed label changed')
        pair=(row,other)
        groups[(row['scenario'],'all','all')].append(pair)
        groups[(row['scenario'],'system',row['system_id'])].append(pair)
        groups[(row['scenario'],'modifier',row['B_modifier'])].append(pair)
        duration='<=20' if row['cycle_min']<=20 else ('21-60' if row['cycle_min']<=60 else '>60')
        groups[(row['scenario'],'cycle_min',duration)].append(pair)
    report=[]
    for (scenario,axis,value),pairs in sorted(groups.items()):
        prior=np.array([a['absolute_error_min'] for a,b in pairs]);current=np.array([b['absolute_error_min'] for a,b in pairs])
        report.append({'scenario':scenario,'group_by':axis,'group':value,'cases':len(pairs),
            'previous_mae_min':float(np.mean(prior)),'current_mae_min':float(np.mean(current)),
            'change_mae_min':float(np.mean(current-prior)),
            'previous_bias_min':float(np.mean([a['predicted_tr_min']-a['observed_tr_min'] for a,b in pairs])),
            'current_bias_min':float(np.mean([b['predicted_tr_min']-b['observed_tr_min'] for a,b in pairs])),
            'current_p90_error_min':float(np.quantile(current,.9))})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('reports/mcmrt_transfer'))
    p.add_argument('--candidate',type=Path)
    p.add_argument('--out',type=Path,default=Path('reports/mcmrt_error_audit'))
    a=p.parse_args();result=audit(a.baseline,a.candidate);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Разбор ошибок переноса MCMRT','',
        'Совпадение строк, измерений, разбиений, временной сетки и гиперпараметров проверено при парном сравнении. Версии дополнительных признаков и целевой переменной следует сверять отдельно в планах.',
        'Проверочные группы уже изучались при разработке: результат не заменяет новую внешнюю валидацию.', '',
        '| Сценарий / группа | Было, мин | Стало, мин | Изменение, мин |','|---|---:|---:|---:|']
    for r in result:
        if r['group_by'] in ('all','cycle_min') or (r['group_by']=='system' and r['group'] in ('mcmrt:01','mcmrt:16')):
            lines.append(f"| {r['scenario']} / {r['group_by']} {r['group']} | {r['previous_mae_min']:.3f} | {r['current_mae_min']:.3f} | {r['change_mae_min']:+.3f} |")
    lines+=['','Положительное изменение означает ухудшение. Все методы сохранены в JSON, включая ухудшившиеся.',
            'Переход к tR / времени первого максимума органики — эмпирическая нормализация, а не физическая модель градиента. Промывка и возврат после первого максимума не задают масштаб.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([r for r in result if r['group_by']=='all'],indent=2))
