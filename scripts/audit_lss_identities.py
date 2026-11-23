"""Cross-check source names with cached PubChem responses; never silently fix typos."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from docx import Document
from rdkit import Chem


def decide(name, response, local):
    props=response.get("PropertyTable",{}).get("Properties",[])
    base={"name":name,"status":"pubchem_unresolved","smiles":None,"full_inchikey":None,
          "source_identity_reviewed":False}
    if len(props)!=1:
        if len(props)>1:base["status"]="multiple_pubchem_structures"
        return base
    p=props[0];smiles=p.get("SMILES") or p.get("IsomericSMILES")
    mol=Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or Chem.MolToInchiKey(mol)!=p.get("InChIKey"):
        return {**base,"status":"pubchem_structure_key_mismatch"}
    key=p["InChIKey"]
    old=local.get("full_inchikey")
    if old and old!=key:
        return {**base,"status":"local_pubchem_identity_conflict","local_inchikey":old,
                "pubchem_inchikey":key,"pubchem_cid":p["CID"]}
    return {**base,"status":"pubchem_name_match_source_identity_unverified",
            "smiles":smiles,"full_inchikey":key,"pubchem_cid":p["CID"],
            "agrees_with_previous_local_match":bool(old)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","build"])
    p.add_argument("--root",type=Path,default=Path("data/external/horner_retention"))
    p.add_argument("--labels",type=Path,default=Path("data/processed/lss_research/parameter_labels.jsonl"))
    p.add_argument("--out",type=Path,default=Path("data/processed/lss_identity_audited"))
    p.add_argument("--report",type=Path,default=Path("reports/lss_identity_audit"))
    a=p.parse_args();cache=a.root/"pubchem";cache.mkdir(parents=True,exist_ok=True)
    doc=Document(a.root/"Compound List and Conditions 2.docx")
    names={int(r.cells[0].text.strip()):r.cells[1].text.strip() for r in doc.tables[0].rows[1:]}
    requests={str(cid):{"name":name,"url":"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"+
                       quote(name,safe="")+"/property/SMILES,InChIKey/JSON"} for cid,name in names.items()}
    if a.command=="prepare":
        (cache/"requests.json").write_text(json.dumps(requests,indent=2))
        config=[]
        for cid,r in requests.items():
            config += ['url = '+json.dumps(r["url"]),'output = '+json.dumps(str((cache/(cid+'.json')).resolve())),
                       'silent','show-error','location','max-time = 25','next']
        (cache/"requests.curl").write_text("\n".join(config[:-1])+"\n")
        print("Prepared",len(requests),"exact source names; no typo corrections")
        return
    rows=[json.loads(l) for l in a.labels.read_text().splitlines()]
    local={r["compound_number"]:r["identity"] for r in rows}
    identities={}
    for cid,name in names.items():
        path=cache/f"{cid}.json"
        try:response=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):response={}
        identity=decide(name,response,local.get(cid,{}))
        identity.update(source_url=requests[str(cid)]["url"],
                        response_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
        identities[cid]=identity
    for row in rows:row["identity"]=identities[row["compound_number"]]
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"parameter_labels.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    summary={"status_counts":dict(Counter(r["status"] for r in identities.values())),
             "usable_compounds":sum(bool(r["smiles"]) for r in identities.values()),
             "usable_series":sum(bool(r["identity"]["smiles"]) for r in rows),
             "identities":identities,"source_identity_reviewed":False,
             "warning":"PubChem resolves a name, not the identity of the authors' actual vial. Conflicts are excluded."}
    (a.out/"identity_audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    a.report.mkdir(parents=True,exist_ok=True)
    (a.report/"audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Проверка названий Horner через PubChem", "",
           f"Сопоставлено {summary['usable_compounds']} из {len(names)} исходных названий; пригодны для исследовательского обучения {summary['usable_series']} серий.",
           "Проверено соответствие SMILES и полного InChIKey средствами RDKit. Совпадение имени в PubChem не подтверждает состав образца автора, соль или стереоизомер экспериментального вещества.",
           "Ошибки в исходных названиях автоматически не исправлялись. Неоднозначные ответы и конфликты с прежней структурой исключаются.","",
           "| Номер | Название | Статус исключения |","|---|---|---|"]
    lines += [f"| {cid} | {r['name']} | {r['status']} |" for cid,r in identities.items() if not r["smiles"]]
    lines += ["", "Все ответы сохранены с URL и SHA-256 в audit.json. Исходная версия обучающих меток сохранена отдельно."]
    (a.report/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!='identities'},indent=2))


if __name__=="__main__":main()
