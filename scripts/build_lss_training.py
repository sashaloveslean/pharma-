"""Build research LSS parameter labels from Horner's published k measurements.

Author-reported k is used directly; t0/tR are not reconstructed. Fixed-temperature
series are fit separately. Entire compositions, including replicates, are held out.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from docx import Document
import numpy as np
from rdkit import Chem

SOURCE = "https://data.mendeley.com/datasets/x292z4gd25/3"


def fit_series(rows):
    grouped = defaultdict(list)
    for r in rows:
        if not math.isfinite(r["phi"]) or not 0 <= r["phi"] <= 1 or not math.isfinite(r["k"]) or r["k"] <= 0:
            raise ValueError("Invalid volume fraction or published k")
        grouped[r["phi"]].append(math.log10(r["k"]))
    phi = sorted(grouped)
    if len(phi) < 5:
        return None
    # Deterministic interpolation check; no choosing the easiest compositions.
    held = phi[1:-1:3]
    train = [p for p in phi if p not in held]
    if len(train) < 3:
        return None
    slope, intercept = np.polyfit(train,[np.mean(grouped[p]) for p in train],1)
    if slope >= 0:
        return None
    checks = [{"phi": p, "observed_log10_k": float(np.mean(grouped[p])),
               "predicted_log10_k": float(intercept+slope*p), "replicates": len(grouped[p])} for p in held]
    return {"log10_kw": float(intercept), "S": float(-slope),
            "calibration_phi": train, "validation_phi": held,
            "calibration_rows": sum(len(grouped[p]) for p in train),
            "validation_rows": sum(len(grouped[p]) for p in held),
            "phi_domain": [min(phi),max(phi)], "validation": checks,
            "validation_mae_log10_k": float(np.mean([abs(c["observed_log10_k"]-c["predicted_log10_k"]) for c in checks])),
            "label_origin": "fitted_from_author_reported_experimental_k",
            "kw_is_extrapolated_to_water": min(phi)>0, "production_approved": False}


def local_identities(root):
    candidates = defaultdict(dict)
    for path in root.glob("*/observations.jsonl"):
        for line in path.open():
            r = json.loads(line)
            if r.get("name") and r.get("smiles") and r.get("compound_id"):
                candidates[r["name"].strip().casefold()][r["compound_id"]] = r["smiles"]
    return candidates


def build(root, corpus, out):
    metadata = json.loads((root/"source_metadata.json").read_text())
    if metadata["version"] != 3 or metadata["data_licence"]["short_name"] != "CC BY 4.0":
        raise ValueError("Unexpected dataset version/license")
    hashes = {}
    for f in metadata["files"]:
        digest = hashlib.sha256((root/f["filename"]).read_bytes()).hexdigest()
        if digest != f["content_details"]["sha256_hash"]:
            raise ValueError("Source checksum mismatch: "+f["filename"])
        hashes[f["filename"]] = digest
    doc = Document(root/"Compound List and Conditions 2.docx")
    names = {int(r.cells[0].text.strip()):r.cells[1].text.strip() for r in doc.tables[0].rows[1:]}
    identities = local_identities(corpus)
    series, observations = defaultdict(list), []
    for i,raw in enumerate(csv.reader((root/"RetentionData").open())):
        cid,w,phi,temp,k = map(float,raw)
        if int(cid) != cid or int(cid) not in names or not all(math.isfinite(x) for x in (w,phi,temp,k)):
            raise ValueError("Invalid source row")
        r = {"observation_id": f"horner:{i+1}", "compound_number": int(cid),
             "name": names[int(cid)], "acn_mass_fraction": w, "phi": phi,
             "temperature_c": temp, "k": k, "source_url": SOURCE,
             "label_origin": "experiment", "k_provenance": "author_reported; t0 derivation not independently audited"}
        observations.append(r); series[(int(cid),temp)].append(r)
    labels = []
    for (cid,temp),rows in sorted(series.items()):
        result = fit_series(rows)
        if result is None:
            continue
        possible = identities.get(names[cid].casefold(),{})
        identity = {"status": "unresolved", "smiles": None, "full_inchikey": None}
        if len(possible) > 1:
            identity["status"] = "ambiguous_local_name_match"
        elif len(possible) == 1:
            key,smiles = next(iter(possible.items()))
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and Chem.MolToInchiKey(mol) == key:
                identity = {"status": "automatic_exact_name_match_needs_source_identity_review",
                            "smiles": smiles, "full_inchikey": key}
        labels.append({"series_id":f"horner:{cid}:{temp:g}", "compound_number":cid, "name":names[cid],
                       "temperature_c":temp, "identity":identity, **result})
    context = {"column":"Waters Acquity BEH C18", "length_mm":50,"id_mm":2.1,"particle_um":1.7,
               "flow_ml_min":.35,"modifier":"acetonitrile",
               "aqueous_recipe":"10 mM aqueous phosphoric acid; mixtures prepared by weight",
               "pH":None,"source_url":SOURCE,"license":"CC BY 4.0"}
    counts = {"observations":len(observations),"compounds":len(names),"temperature_series":len(series),
              "fitted_series":len(labels),"structure_linked_series":sum(bool(x["identity"]["smiles"]) for x in labels),
              "structure_linked_compounds":len({x["compound_number"] for x in labels if x["identity"]["smiles"]}),
              "median_series_validation_mae_log10_k":float(np.median([x["validation_mae_log10_k"] for x in labels])),
              "source_hashes":hashes,"context":context,"production_approved":False,
              "limitations":["One column and modifier; cannot learn column transfer.",
                             "Automatic name matching is provisional, ambiguous matches excluded.",
                             "Published k, not reconstructed or measured tR/t0 in this import.",
                             "LSS is an approximation; held-out compositions evaluate interpolation only."]}
    out.mkdir(parents=True,exist_ok=True)
    for name,records in [("k_observations.jsonl",observations),("parameter_labels.jsonl",labels)]:
        (out/name).write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in records))
    (out/"summary.json").write_text(json.dumps(counts,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return counts


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("data/external/horner_retention"))
    p.add_argument("--corpus",type=Path,default=Path("data/processed/public_retention"))
    p.add_argument("--out",type=Path,default=Path("data/processed/lss_research"))
    args=p.parse_args()
    print(json.dumps(build(args.root,args.corpus,args.out),indent=2))
