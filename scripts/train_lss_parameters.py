"""Research-only molecule/temperature -> LSS parameters on one fixed column.

Group-held-out molecules evaluate the end-to-end predicted log(k), not just fitted
parameter labels. No TFV/FTC recommendations or production model replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import GroupKFold

from train_open_retention import molecular


def molecule_folds(groups, reference_groups=None):
    """Freeze legacy test assignments while assigning new molecules once by hash."""
    reference=np.array(reference_groups if reference_groups is not None else groups)
    assigned={}
    for fold,(_,test) in enumerate(GroupKFold(n_splits=5).split(reference,groups=reference)):
        for g in reference[test]:assigned[str(g)]=fold
    for g in groups:
        if str(g) not in assigned:
            assigned[str(g)]=int(hashlib.sha256(str(g).encode()).hexdigest(),16)%5
    for fold in range(5):
        test=np.array([i for i,g in enumerate(groups) if assigned[str(g)]==fold],dtype=int)
        fit=np.array([i for i,g in enumerate(groups) if assigned[str(g)]!=fold],dtype=int)
        if len(test) and len(fit):yield fold,fit,test


def train(path,out,split_reference=None,baseline_predictions=None):
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    rows=[r for r in rows if r["identity"]["smiles"]]
    groups=np.array([r["identity"]["full_inchikey"].split("-")[0] for r in rows])
    if len(set(groups))<10:
        raise ValueError("Too few molecular groups for the research comparison")
    X=np.array([molecular(r["identity"]["smiles"])+[r["temperature_c"]] for r in rows])
    y=np.array([[r["log10_kw"],r["S"]] for r in rows])
    params=dict(loss_function="MultiRMSE",iterations=250,depth=4,learning_rate=.04,
                random_seed=42,verbose=False,allow_writing_files=False,thread_count=4)
    predictions=[]
    reference_groups=None
    if split_reference:
        old=[json.loads(l) for l in split_reference.read_text().splitlines()]
        reference_groups=[r["identity"]["full_inchikey"].split("-")[0] for r in old if r["identity"]["smiles"]]
    for fold,fit,test in molecule_folds(groups,reference_groups):
        model=CatBoostRegressor(**params);model.fit(X[fit],y[fit])
        pred=model.predict(X[test])
        for idx,estimate in zip(test,pred):
            r=rows[idx]
            same_temp=[i for i in fit if rows[i]["temperature_c"]==r["temperature_c"]]
            baseline=np.median(y[same_temp if same_temp else fit],axis=0)
            for c in r["validation"]:
                actual=c["observed_log10_k"]
                value=float(estimate[0]-estimate[1]*c["phi"])
                naive=float(baseline[0]-baseline[1]*c["phi"])
                predictions.append({"fold":fold,"series_id":r["series_id"],"name":r["name"],
                    "molecular_group":groups[idx],"phi":c["phi"],"observed_log10_k":actual,
                    "predicted_log10_k":value,"baseline_log10_k":naive,
                    "absolute_error_log10_k":abs(value-actual),"baseline_error_log10_k":abs(naive-actual),
                    "predicted_S_positive":bool(estimate[1]>0)})
    mae=float(np.mean([r["absolute_error_log10_k"] for r in predictions]))
    baseline=float(np.mean([r["baseline_error_log10_k"] for r in predictions]))
    group_mae={str(g):float(np.mean([r["absolute_error_log10_k"] for r in predictions if r["molecular_group"]==g])) for g in sorted(set(groups))}
    report={"status":"research_only_not_promoted","series":len(rows),"molecular_groups":len(set(groups)),
            "evaluation":"5 folds grouped by molecular connectivity; all temperatures of a molecule held together",
            "parameter_label_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "split_reference_sha256":hashlib.sha256(split_reference.read_bytes()).hexdigest() if split_reference else None,
            "target":["log10_kw","S"],"held_out_composition_predictions":len(predictions),
            "mae_log10_k":mae,"baseline_mae_log10_k":baseline,
            "macro_molecule_mae_log10_k":float(np.mean(list(group_mae.values()))),
            "median_absolute_error_log10_k":float(np.median([r["absolute_error_log10_k"] for r in predictions])),
            "beats_baseline_in_this_experiment":mae<baseline,"production_approved":False,
            "limitations":["Provisional automatic name-to-structure matches, not fully curated identities.",
                "One BEH C18 column, ACN/phosphoric acid only: no transfer to Acclaim/MeOH demonstrated.",
                "No TFV/FTC validation and no peak widths or Rs.",
                "Published k values; unknown t0 methodology and pH not invented.",
                "First research cross-validation, not a separate final validation dataset."]}
    if baseline_predictions:
        prior=[json.loads(l) for l in baseline_predictions.read_text().splitlines()]
        key=lambda r:(r["molecular_group"],r["series_id"],r["phi"],r["fold"])
        previous={key(r):r for r in prior}
        common=[r for r in predictions if key(r) in previous]
        report["same_cases_comparison"]={"cases":len(common),
            "previous_mae_log10_k":float(np.mean([previous[key(r)]["absolute_error_log10_k"] for r in common])) if common else None,
            "current_mae_log10_k":float(np.mean([r["absolute_error_log10_k"] for r in common])) if common else None,
            "baseline_predictions_sha256":hashlib.sha256(baseline_predictions.read_bytes()).hexdigest(),
            "note":"Same molecules, compositions and test folds; data expansion and identity filtering both changed."}
    out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in predictions))
    model=CatBoostRegressor(**params);model.fit(X,y);model.save_model(str(out/"research_lss.cbm"))
    examples=[]
    for name in ["caffeine","methylparaben","ibuprofen","ketoprofen"]:
        subset=[r for r in predictions if r["name"]==name]
        if subset:
            examples.append(f"| {name} | {len(subset)} | {np.mean([r['absolute_error_log10_k'] for r in subset]):.3f} |")
    text=["# Прогноз параметров физической модели: первый эксперимент", "",
          f"Использованы {len(rows)} температурных серий, {len(set(groups))} групп молекул с предварительным сопоставлением структур.",
          "Все температуры одной молекулы исключались из обучения вместе. Проверяем log10(k) на отложенных составах для невиденных молекул.","",
          f"MAE log10(k): **{mae:.3f}**. Простой ориентир (медианные параметры обучающих веществ при той же температуре): **{baseline:.3f}**.",
          f"Средняя ошибка с равным весом молекул: {report['macro_molecule_mae_log10_k']:.3f}.",
          "Ошибка 0.3 в log10(k) означает примерно двукратное расхождение k; это не минуты и не ошибка tR.","",
          "| Вещество | Проверочные составы × температуры | MAE log10(k) |","|---|---:|---:|",*examples,"",
          "Результат исследовательский: структуры требуют проверки, колонка и растворитель только одни. Перенос на выбранный метод TFV/FTC не проверен. Основная модель и выбранный градиент не изменены."]
    (out/"REPORT.md").write_text("\n".join(text)+"\n")
    if "same_cases_comparison" in report:
        c=report["same_cases_comparison"]
        with (out/"REPORT.md").open("a") as f:
            f.write(f"\nСопоставимая проверка: {c['cases']} одинаковых случаев с сохранёнными тестовыми группами. Предыдущая MAE: {c['previous_mae_log10_k']}; новая: {c['current_mae_log10_k']}. Одновременно изменились охват обучения и фильтрация структур.\n")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/processed/lss_research/parameter_labels.jsonl"))
    p.add_argument("--out",type=Path,default=Path("reports/lss_parameter_learning"))
    p.add_argument("--split-reference",type=Path)
    p.add_argument("--baseline-predictions",type=Path)
    a=p.parse_args()
    if a.baseline_predictions and not a.split_reference:p.error("Baseline comparison requires frozen reference splits")
    print(json.dumps(train(a.data,a.out,a.split_reference,a.baseline_predictions),indent=2))
