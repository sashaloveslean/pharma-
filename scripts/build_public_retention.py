#!/usr/bin/env python3
"""Build separate, traceable METLIN and U Liege measurement datasets."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

from rdkit import Chem, rdBase
from open_retention import api, number


def write_dataset(folder, rows, systems, issues):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "observations.jsonl").open("w") as file:
        for r in rows:
            file.write(json.dumps(r) + "\n")
    (folder / "systems.json").write_text(json.dumps(systems, indent=2))
    report = {"rows": len(rows), "unique_molecules": len({r["compound_id"] for r in rows}),
              "systems": len(systems), "issues": dict(issues),
              "missing_peak_width": sum(r.get("peak_width") is None for r in rows),
              "missing_run_id": sum(r.get("run_id") is None for r in rows)}
    (folder / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(folder.name, json.dumps(report), flush=True)


def smrt(root, out):
    folder = root / "metlin_smrt"
    metadata = json.loads((folder / "source_metadata.json").read_text())
    path = folder / "SMRT_dataset.csv"
    revision = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, issues = [], Counter()
    with path.open() as file:
        for index, raw in enumerate(csv.DictReader(file, delimiter=";")):
            with rdBase.BlockLogs():
                mol = Chem.MolFromInchi(raw["inchi"])
            rt = number(raw["rt"])
            if mol is None or rt is None or rt <= 0:
                issues["invalid_inchi_or_rt"] += 1
                continue
            rows.append({"observation_id": f"smrt:{index}", "compound_id": Chem.MolToInchiKey(mol),
                         "smiles": Chem.MolToSmiles(mol, isomericSmiles=True), "cid": raw["pubchem"],
                         "name": None, "system_id": "smrt:RP", "run_id": None,
                         "retention_time_min": rt / 60, "original_rt": rt, "original_rt_unit": "s",
                         "rt_unit": "min", "source": "METLIN SMRT", "source_revision": revision,
                         "source_url": metadata["url"], "license": metadata["license"],
                         "peak_width": None, "Rs": None})
            if (index + 1) % 10000 == 0:
                print(f"SMRT parsed {index + 1}", flush=True)
    systems = {"smrt:RP": {"metadata": {}, "gradient_program": [],
                           "source_info": [{"method.type": "RP", "name": "METLIN SMRT single method"}],
                           "metadata_status": "structure-only baseline; method conditions not yet mapped"}}
    write_dataset(out / "metlin_smrt", rows, systems, issues)


def uliege(root, out, resolve):
    import openpyxl
    folder = root / "uliege_csm5gsmr5t"
    path = folder / "Summary.xlsx"
    sheet = iter(openpyxl.load_workbook(path, read_only=True, data_only=True).active.values)
    headers = next(sheet)
    raw = [dict(zip(headers, r)) for r in sheet if r[0] is not None]
    cache_path = folder / "structure_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if resolve:
        for name in sorted({r["compound"] for r in raw}):
            if name in cache and cache[name].get("status") == "resolved":
                continue
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/" + quote(name, safe="")
                   + "/property/SMILES,InChIKey,IUPACName/JSON")
            try:
                result = api(url)["PropertyTable"]["Properties"]
                if len(result) != 1 or not result[0].get("SMILES"):
                    raise ValueError("Ambiguous name or missing structure")
                cache[name] = {"status": "resolved", "properties": result[0], "source_url": url,
                               "needs_identity_review": True}
            except Exception as exc:
                cache[name] = {"status": "unresolved", "error": str(exc), "source_url": url}
            cache_path.write_text(json.dumps(cache, indent=2))
            print("PubChem", name, cache[name]["status"], flush=True)
            time.sleep(0.25)
    issues, systems, rows = Counter(), {}, []
    revision = hashlib.sha256(path.read_bytes()).hexdigest()
    for r in raw:
        identity = cache.get(r["compound"], {})
        if identity.get("status") != "resolved":
            issues["unresolved_structure"] += 1
            continue
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(identity["properties"]["SMILES"])
        rt = number(r["retention_time_corrected"])
        if mol is None or rt is None or rt <= 0:
            issues["invalid_structure_or_rt"] += 1
            continue
        # Physical system group stays stable across pH and gradients for holdout tests.
        sid = f"uliege:{r['system']}:{r['column']}"
        meta = {"column.name": "Waters XSelect HSS T3", "column.usp.code": "L1",
                "column.length": 100, "column.id": 2.1, "column.particle.size": 3.5,
                "column.temperature": 25, "column.flowrate": 0.3,
                "eluent.A.pH": r["pH_measured"], "eluent.A.h2o": 100,
                "eluent.B.meoh": 100, "gradient.time_min": r["gradient_time"]}
        systems.setdefault(sid, {"metadata": {}, "gradient_program": [],
                                 "source_info": [{"method.type": "RP"}]})
        rows.append({"observation_id": f"uliege:{r['injection_id']}",
                     "compound_id": Chem.MolToInchiKey(mol), "smiles": Chem.MolToSmiles(mol),
                     "name": r["compound"], "system_id": sid, "run_id": None,
                     "retention_time_min": rt, "rt_unit": "min", "metadata": meta,
                     "gradient_program": [], "gradient_status": "nominal duration only; shortened methods need raw-file linkage",
                     "original_retention_time_min": r["retention_time"],
                     "correction_retention_time_min": r["correction_retention_time"],
                     "experimental_record": r, "source": "U Liege", "source_revision": revision,
                     "source_url": "https://data.mendeley.com/datasets/csm5gsmr5t/1",
                     "method_source_url": "https://doi.org/10.1016/j.dib.2022.108017",
                     "identity_source": identity, "identity_status": "automatic_name_match_needs_review",
                     "license": "CC-BY-4.0", "peak_width": None, "Rs": None})
    # Keep unresolved observations in full, instead of silently losing measurements.
    dest = out / "uliege"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "unresolved.json").write_text(json.dumps([r for r in raw if cache.get(r["compound"], {}).get("status") != "resolved"], indent=2))
    write_dataset(dest, rows, systems, issues)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/external"))
    p.add_argument("--out", type=Path, default=Path("data/processed/public_retention"))
    p.add_argument("--source", choices=["all", "smrt", "uliege"], default="all")
    p.add_argument("--resolve-names", action="store_true")
    args = p.parse_args()
    if args.source in {"all", "smrt"}:
        smrt(args.root, args.out)
    if args.source in {"all", "uliege"}:
        uliege(args.root, args.out, args.resolve_names)


if __name__ == "__main__":
    main()
