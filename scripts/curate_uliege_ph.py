"""Recover row-level measured pH; retain provenance and block premature training."""
import json
import hashlib
from pathlib import Path
from collections import Counter
import openpyxl

OUT=Path('reports/uliege_ph_recovery')
ARTICLE='https://doi.org/10.1016/j.dib.2022.108017'
RECIPES={2:dict(nominal_ph=2.7,components=['formic acid']),
         3.5:dict(nominal_ph=3.5,components=['ammonium formate','formic acid']),
         5:dict(nominal_ph=5.,components=['ammonium acetate','acetic acid']),
         6.5:dict(nominal_ph=6.5,components=['ammonium bicarbonate','formic acid']),
         8:dict(nominal_ph=8.,components=['ammonium bicarbonate'])}

def annotate(record):
    label=record['pH_target']
    recipe=RECIPES.get(label)
    return dict(record,phase_recipe=recipe,reported_buffer_concentration_mM=10 if recipe else None,
        concentration_basis='article states buffer 10 mM; individual component concentrations not resolved',
        pH_label_is_measured=False,measured_pH=record['pH_measured'],
        phase_source=ARTICLE,phase_source_location='Table 1; section 2.1',
        gradient_status='nominal duration/slope only; full vs shortened program unresolved per injection',
        pH_effect_interpretation='joint pH and phase-composition change, not isolated pH effect',
        is_training_ready=False)

def run():
    path=Path('data/external/uliege_csm5gsmr5t/Summary.xlsx')
    it=iter(openpyxl.load_workbook(path,read_only=True,data_only=True).active.values)
    headers=next(it);raw=[dict(zip(headers,r)) for r in it if r[0] is not None]
    imported={json.loads(l)['experimental_record']['injection_id']:json.loads(l)
              for l in Path('data/processed/public_retention/uliege/observations.jsonl').read_text().splitlines()}
    records=[]
    for r in raw:
        item=annotate(r);old=imported.get(r['injection_id'])
        item.update(source_file=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_dataset='https://data.mendeley.com/datasets/csm5gsmr5t/1',license='CC BY 4.0',
            compound_id=old['compound_id'] if old else None,
            smiles=old['smiles'] if old else None,
            identity_status=old.get('identity_status') if old else 'unresolved')
        records.append(item)
    nonqc=[r for r in records if r['is_QC']==0]
    count=Counter(str(r['pH_target']) for r in nonqc)
    summary=dict(raw_rows=len(raw),qc_rows=sum(r['is_QC']==1 for r in records),non_qc_rows=len(nonqc),
        non_qc_compound_names=len({r['compound'] for r in nonqc}),non_qc_by_ph_label=dict(count),
        non_qc_with_measured_ph=sum(isinstance(r['measured_pH'],(int,float)) for r in nonqc),
        non_qc_resolved_structure=sum(r['smiles'] is not None for r in nonqc),
        systems=sorted({r['system'] for r in records}),
        nominal_gradient_durations=sorted({r['gradient_time'] for r in records}),
        training_ready_rows=0,production_updated=False,
        independent_of_previous_paired_study=True,
        not_newly_downloaded=True,
        original_pair_recipes_status='MultiConditionRT supplementary recipes still unverified')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'observations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'phase_recipes.json').write_text(json.dumps(RECIPES,indent=2)+'\n')
    text=f'''# Восстановление построчных pH U Liège

Предыдущий аудит systems.json пропустил pH в отдельных observations.jsonl. Это исправление покрытия, не новая загрузка.

Локальная Summary.xlsx: {len(raw)} строк; QC: {summary['qc_rows']}; остальные: {len(nonqc)}. Имен веществ без QC: {summary['non_qc_compound_names']}. С измеренным pH: {summary['non_qc_with_measured_ph']}; с автоматически сопоставленной структурой: {summary['non_qc_resolved_structure']}.

Число строк без QC по исходной метке pH: {dict(count)}.

[Первоисточник]({ARTICLE}), таблица 1 и разделы 2.1–2.2: пять составов фаз и заявленная концентрация буфера 10 мМ. Метка 2 соответствует номинальному pH 2,7; фактический pH сохранён отдельно. 10 мМ не приравнены автоматически к концентрации отдельной соли. При смене pH меняется и состав фазы. В источнике есть укороченные градиенты, поэтому номинальные 20/60 минут не восстановлены как полные программы каждого ввода.

## Готовность

Обучение пока запрещено для всех строк: требуется связать каждый ввод с фактической программой, проверить автоматические структуры и разделить QC/аналитические измерения. Повторные вводы не являются независимыми молекулами. Разбиение должно исключать одну молекулу со всеми повторами и учитывать систему/серию.

Это другая публикация относительно предыдущего эксперимента MultiConditionRT, но не новый внешний тест для всего проекта: файл уже был импортирован ранее. Составы четырёх прежних пар pH 5/8 по приложению MultiConditionRT всё ещё не подтверждены. Рабочая модель не изменена.
'''
    (OUT/'REPORT.md').write_text(text)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':run()
