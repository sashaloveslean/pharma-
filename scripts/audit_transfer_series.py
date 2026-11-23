"""Identify comparable public gradient series; no missing timings are imputed."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def audit(root):
    systems=json.loads((root/"systems.json").read_text())
    grouped=defaultdict(list)
    for sid,s in systems.items():
        info=s.get("source_info",[]);m=s["metadata"];p=s["gradient_program"]
        if not info or info[0].get("method.type")!="RP" or len(p)<2:continue
        source=tuple(sorted(i.get("url","") for i in info))
        if not any(source):continue
        # Equal recorded metadata is a candidate, not proof of equal instruments.
        context={k:v for k,v in m.items() if k.startswith(("column.","eluent.")) and k!="column.t0"}
        grouped[(json.dumps(context,sort_keys=True),source)].append(sid)
    groups=[]
    memberships=defaultdict(list)
    for (context,source),ids in grouped.items():
        signatures={}
        for sid in ids:
            p=systems[sid]["gradient_program"]
            try:
                normalized=[{k:float(v) for k,v in row.items()} for row in p]
                signature=json.dumps(normalized,sort_keys=True)
            except (ValueError,TypeError):continue
            signatures[sid]=signature
        if len(set(signatures.values()))<2:continue
        m=json.loads(context)
        group={"group_id":hashlib.sha256(json.dumps([context,source]).encode()).hexdigest()[:16],
               "systems":ids,"source_urls":source,"column":m.get("column.name"),
               "modifier_B":{"methanol_percent":m.get("eluent.B.meoh"),"acetonitrile_percent":m.get("eluent.B.acn")},
               "distinct_full_programs":len(set(signatures.values())),"context":m,
               "blockers":["instrument_equivalence_unreviewed","t0_origin_unverified",
                           "dwell_and_postcolumn_times_missing","program_independence_requires_review"],
               "candidate_molecules":[],"training_ready":False}
        if m.get("column.temperature") in (None,""):group["blockers"].append("missing_temperature")
        groups.append(group)
        for sid,signature in signatures.items():memberships[sid].append((len(groups)-1,signature))
    observations=defaultdict(lambda:defaultdict(set))
    with (root/"observations.jsonl").open() as source_file:
        for line in source_file:
            r=json.loads(line)
            for index,signature in memberships.get(r["system_id"],[]):
                observations[index][r["compound_id"]].add(signature)
    for i,g in enumerate(groups):
        g["candidate_molecules"]=[{"full_inchikey":key,"distinct_programs":len(programs)}
                                  for key,programs in observations[i].items() if len(programs)>=3]
        g["molecules_with_at_least_three_programs"]=len(g["candidate_molecules"])
    return {"groups":groups,"source_sha256":hashlib.sha256((root/"systems.json").read_bytes()).hexdigest(),
            "status":"candidates_only_no_parameter_labels_approved"}


if __name__=="__main__":
    result=audit(Path("data/processed/public_retention/report"))
    out=Path("reports/transfer_series_audit");out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Серии для переноса на другие условия", "",
           "Это кандидаты для проверки источников, не готовые параметры физической модели.", "",
           "| Колонка | MeOH в B, % | Разных полных программ | Молекул с ≥3 программами | Пробелы |",
           "|---|---:|---:|---:|---|"]
    for g in result["groups"]:
        lines.append(f"| {g['column']} | {g['modifier_B']['methanol_percent']} | {g['distinct_full_programs']} | {g['molecules_with_at_least_three_programs']} | {', '.join(g['blockers'])} |")
    lines += ["", "Разные полные программы могут различаться только промывкой после выхода пиков. Их нельзя автоматически считать независимыми калибровочными экспериментами.",
              "Положительные/отрицательные режимы MS и копии источника не считаются дополнительными градиентами при одинаковой программе.",
              "Температура, t₀, задержки и равенство приборов не восстанавливались предположениями."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps([{k:g[k] for k in ('column','distinct_full_programs','molecules_with_at_least_three_programs')} for g in result['groups']],indent=2))
