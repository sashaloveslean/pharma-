#!/usr/bin/env python3
"""Fetch licensed OA full texts and preserve traceable HPLC evidence candidates."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

TOPICS = {"retention": r"retention time|\btR\b", "resolution": r"resolution|\bRs\b",
          "width": r"peak width|half.height|FWHM", "tailing": r"tailing|asymmetry",
          "efficiency": r"theoretical plate|plate count", "pressure": r"back.?pressure|\bbar\b|\bpsi\b",
          "conditions": r"mobile phase|gradient|column|buffer|flow rate"}


def extract_evidence(root, pmcid, metadata, url):
    """JATS tables can live in floats-group, outside body; never restrict them to body."""
    snippets, tables = [], []
    for tag in ("p", "table-wrap"):
        nodes = root.findall(".//body//p") if tag == "p" else root.findall(".//table-wrap")
        for index, node in enumerate(nodes):
            text = " ".join(" ".join(node.itertext()).split())
            topics = [name for name, pattern in TOPICS.items() if re.search(pattern, text, re.I)]
            node_id = node.get("id") or f"{tag}:{index}"
            if tag == "table-wrap":
                tables.append({"article": pmcid, "node_id": node_id,
                               "caption": " ".join((node.findtext("label", "") + " " +
                                          " ".join(" ".join(n.itertext()) for n in node.findall("caption"))).split()),
                               "rows": [[{"text": " ".join(" ".join(c.itertext()).split()),
                                          "colspan": c.get("colspan", "1"), "rowspan": c.get("rowspan", "1")}
                                         for c in tr if c.tag in {"td", "th"}] for tr in node.findall(".//tr")],
                               "source_url": url, "reviewed": False, "training_ready": False})
            if topics and re.search(r"\d", text):
                snippets.append({"article": pmcid, "doi": metadata.get("doi"), "source_url": url,
                                 "node_id": node_id, "node_type": tag, "topics": topics, "text": text,
                                 "mentions_target": bool(re.search(r"tenofovir|emtricitabine", text, re.I)),
                                 "reviewed": False, "training_ready": False})
    return snippets, tables


def reindex_cached(out):
    """Rebuild evidence from already licensed local XML; perform no network requests."""
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence = []
    for entry in manifest["articles"]:
        if entry["status"] != "downloaded":
            continue
        path = out / (entry["pmcid"] + ".xml")
        content = path.read_bytes()
        if entry.get("sha256") != hashlib.sha256(content).hexdigest():
            raise ValueError(f"Source checksum mismatch: {path}")
        root = ET.fromstring(content)
        if not license_allowed(root)[0]:
            raise ValueError(f"Source reuse license not confirmed: {path}")
        snippets, tables = extract_evidence(root, entry["pmcid"], entry, entry["url"])
        (out / (entry["pmcid"] + "_tables.json")).write_text(json.dumps(tables, indent=2))
        entry["evidence_candidates"] = len(snippets)
        evidence.extend(snippets)
    manifest["evidence_candidates"] = len(evidence)
    manifest["target_specific_candidates"] = sum(r["mentions_target"] for r in evidence)
    manifest["table_extraction_scope"] = "all article table-wrap nodes, including floats-group"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (out / "evidence.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evidence))
    print(json.dumps({k: v for k, v in manifest.items() if k != "articles"}, indent=2))


def license_allowed(root):
    permissions = root.findall(".//permissions")
    license_text = " ".join(" ".join(p.itertext()) + " " + " ".join(
        str(v) for node in p.iter() for v in node.attrib.values()) for p in permissions)
    if re.search(r"non.?commercial|no.?derivatives|by-nc|by-nd", license_text, re.I):
        return False, license_text
    allowed = bool(re.search(r"creativecommons.org/(?:licenses/by(?:-sa)?/|publicdomain/zero/)", license_text, re.I))
    return allowed, license_text


def main(args):
    if args.reindex_cached:
        return reindex_cached(args.out)
    records = json.loads(args.bibliography.read_text())["records"]
    selected = {r["pmcid"]: r for r in records if r.get("isOpenAccess") == "Y" and r.get("pmcid")
                and re.search(r"chromatograph|hplc|lc.ms|uplc", r.get("title", ""), re.I)}
    args.out.mkdir(parents=True, exist_ok=True)
    inventory, evidence = [], []
    def fetch(item):
        pmcid, metadata = item
        path = args.out / (pmcid + ".xml")
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        if path.exists():
            content = path.read_bytes()
        else:
            content = subprocess.run(["curl", "-sSfL", "--retry", "2", "--max-time", "90", url],
                                     check=True, capture_output=True).stdout
        root = ET.fromstring(content)
        allowed, license_text = license_allowed(root)
        entry = {"pmcid": pmcid, "title": metadata["title"], "doi": metadata.get("doi"),
                 "url": url, "license_evidence": license_text,
                 "status": "downloaded" if allowed else "excluded_license_not_confirmed_for_reuse"}
        if not allowed:
            return entry, []
        path.write_bytes(content)
        entry.update(path=str(path), sha256=hashlib.sha256(content).hexdigest())
        snippets, tables = extract_evidence(root, pmcid, metadata, url)
        (args.out / (pmcid + "_tables.json")).write_text(json.dumps(tables, indent=2))
        entry["evidence_candidates"] = len(snippets)
        return entry, snippets
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, item): item[0] for item in selected.items()}
        for future in as_completed(futures):
            try:
                record, snippets = future.result()
                inventory.append(record)
                evidence.extend(snippets)
                print(record["pmcid"], record["status"], len(snippets), flush=True)
            except Exception as exc:
                inventory.append({"pmcid": futures[future], "status": "failed", "error": str(exc)})
    report = {"searched_articles": len(selected), "articles": sorted(inventory, key=lambda r:r["pmcid"]),
              "evidence_candidates": len(evidence), "target_specific_candidates": sum(r["mentions_target"] for r in evidence),
              "note": "Evidence snippets are not normalized measurements; manual review and method/run linkage required"}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2))
    with (args.out / "evidence.jsonl").open("w") as file:
        for row in evidence:
            file.write(json.dumps(row) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "articles"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bibliography", type=Path, default=Path("data/external/europepmc/tenofovir_emtricitabine_bibliography.json"))
    p.add_argument("--out", type=Path, default=Path("data/external/open_methods"))
    p.add_argument("--reindex-cached", action="store_true", help="Reparse verified licensed local XML without downloading")
    main(p.parse_args())
