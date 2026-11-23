#!/usr/bin/env python3
"""Import versioned RepoRT measurements; no customer data or API keys required."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "https://api.github.com/repos/michaelwitting/RepoRT"


def api(url):
    result = subprocess.run(
        ["curl", "--silent", "--show-error", "--fail", "--retry", "2",
         "--connect-timeout", "15", "--max-time", "90", "-L", url],
        check=True, capture_output=True,
    )
    return json.loads(result.stdout)


def download(root, systems):
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "source": "RepoRT", "revision": api(REPO + "/commits/master")["sha"],
        "license_url": "https://github.com/michaelwitting/RepoRT/blob/master/LICENSE",
        "files": {},
    }
    revision = manifest["revision"]
    for relative in ["LICENSE", "DataDescription.md"] + [
        f"processed_data/{sid}/{sid}_{suffix}" for sid in systems for suffix in
        ("metadata.tsv", "info.tsv", "gradient.tsv", "rtdata_canonical_success.tsv",
         "rtdata_isomeric_success.tsv")
    ]:
        path = root / relative
        if path.exists() and relative in manifest["files"]:
            if hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][relative]["sha256"]:
                continue
        record = api(REPO + f"/contents/{relative}?ref={revision}")
        content = base64.b64decode(record["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["files"][relative] = {
            "sha256": hashlib.sha256(content).hexdigest(), "git_blob": record["sha"],
            "source_url": record["html_url"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Downloaded {relative}", flush=True)


def download_all(root):
    """Discover all systems, downloading only measurement and method tables."""
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "source": "RepoRT", "revision": api(REPO + "/commits/master")["sha"], "files": {}}
    revision = manifest["revision"]
    tree = api(REPO + f"/git/trees/{revision}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    (root / "source_tree.json").write_text(json.dumps(tree))
    suffixes = ("_metadata.tsv", "_info.tsv", "_gradient.tsv",
                "_rtdata_canonical_success.tsv", "_rtdata_isomeric_success.tsv")
    selected = [f for f in tree["tree"] if f["type"] == "blob" and (
        f["path"] in {"LICENSE", "DataDescription.md"} or
        (f["path"].startswith("processed_data/") and f["path"].endswith(suffixes)))]
    def fetch(item):
        relative = item["path"]
        path = root / relative
        previous = manifest["files"].get(relative)
        if previous and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == previous["sha256"]:
            return relative, previous
        url = f"https://raw.githubusercontent.com/michaelwitting/RepoRT/{revision}/{relative}"
        content = subprocess.run(["curl", "-sS", "-fL", "--retry", "2", "--max-time", "90", url],
                                 capture_output=True, check=True).stdout
        # Verify bytes against Git's content-addressed blob, including its header.
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if blob != item["sha"]:
            raise ValueError(f"Git blob mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return relative, {"sha256": hashlib.sha256(content).hexdigest(), "git_blob": blob,
                          "source_url": f"https://github.com/michaelwitting/RepoRT/blob/{revision}/{relative}"}
    failures = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, item): item["path"] for item in selected}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                relative, record = future.result()
                manifest["files"][relative] = record
            except Exception as exc:
                failures.append({"path": futures[future], "error": str(exc)})
            if i % 50 == 0 or i == len(futures):
                manifest["failures"] = failures
                manifest_path.write_text(json.dumps(manifest, indent=2))
                print(f"RepoRT {i}/{len(futures)}, failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} files failed; rerun resumes verified files")


def table(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")), delimiter="\t"))


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def load_observations(path):
    systems_path = path.parent / "systems.json"
    systems = json.loads(systems_path.read_text()) if systems_path.exists() else {}
    rows = []
    with path.open() as file:
        for line in file:
            row = json.loads(line)
            if row.get("label_origin", "experiment") != "experiment":
                raise ValueError("Public experimental dataset cannot include synthetic or predicted labels")
            rows.append({**systems.get(row["system_id"], {}), **row})
    return rows


def build(root, out, compact=False):
    from rdkit import Chem, rdBase
    from collections import Counter

    manifest = json.loads((root / "manifest.json").read_text())
    rows, issues, systems = [], Counter(), {}
    for folder in sorted((root / "processed_data").iterdir()):
        sid = folder.name
        metadata = table(folder / f"{sid}_metadata.tsv")
        if len(metadata) != 1:
            raise ValueError(f"Expected one metadata record: {sid}")
        meta = metadata[0]
        # RepoRT uses 0 for unreported pH; it is not a measured pH of zero.
        for key in meta:
            if key.endswith(".pH") and number(meta[key]) == 0:
                meta[key] = None
        gradient = table(folder / f"{sid}_gradient.tsv")
        info = table(folder / f"{sid}_info.tsv")
        systems["report:" + sid] = {"metadata": meta, "gradient_program": gradient, "source_info": info}
        records = {}
        for path in sorted(folder.glob(f"{sid}_rtdata_*_success.tsv")):
            for raw in table(path):
                records[raw["id"]] = (raw, path)
        seen = set()
        for raw, path in records.values():
            smiles = raw.get("smiles.std") or raw.get("pubchem.smiles.isomeric") or raw.get("pubchem.smiles.canonical")
            rt = number(raw.get("rt"))
            with rdBase.BlockLogs():
                mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None or rt is None or rt <= 0:
                issues["invalid_structure_or_rt"] += 1
                continue
            # Preserve salts/prodrugs and stereochemistry; do not neutralize labels.
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            key = Chem.MolToInchiKey(mol)
            duplicate = (key, rt)
            if duplicate in seen:
                issues["duplicate_same_structure_rt_system"] += 1
            seen.add(duplicate)
            rows.append({
                "observation_id": "report:" + raw["id"], "compound_id": key,
                "smiles": smiles, "name": raw.get("name"),
                "structure_representation": "isomeric" if "isomeric" in path.name else "canonical",
                "system_id": "report:" + sid, "run_id": None,
                "retention_time_min": rt, "rt_unit": "min",
                "metadata": meta, "gradient_program": gradient, "source_info": info,
                "source": "RepoRT", "source_revision": manifest["revision"],
                "source_url": manifest["files"][str(path.relative_to(root))]["source_url"],
                "license": "CC-BY-SA-4.0", "peak_width": None, "Rs": None,
            })
    if not rows:
        raise ValueError("No usable observations")
    out.mkdir(parents=True, exist_ok=True)
    (out / "systems.json").write_text(json.dumps(systems, indent=2))
    with (out / "observations.jsonl").open("w") as file:
        for row in rows:
            file.write(json.dumps({k: v for k, v in row.items() if not compact or
                                   k not in {"metadata", "gradient_program", "source_info"}}) + "\n")
    report = {
        "rows": len(rows), "unique_molecules": len({r["compound_id"] for r in rows}),
        "systems": dict(Counter(r["system_id"] for r in rows)), "issues": dict(issues),
        "missing_peak_width": len(rows), "missing_confirmed_run_id": len(rows),
        "limitation": "System membership does not establish co-injection; Rs cannot be inferred.",
        "source_revision": manifest["revision"],
    }
    (out / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["download", "download-all", "build"])
    p.add_argument("--systems", nargs="+", default=["0001", "0002", "0003", "0004", "0005", "0006"])
    p.add_argument("--root", type=Path, default=Path("data/external/report"))
    p.add_argument("--out", type=Path, default=Path("data/processed/open_retention"))
    p.add_argument("--compact", action="store_true", help="Store method metadata once in systems.json")
    args = p.parse_args()
    if any(len(s) != 4 or not s.isdigit() for s in args.systems):
        p.error("System IDs must be four digits")
    if args.command == "download-all":
        download_all(args.root)
    elif args.command == "download":
        download(args.root, args.systems)
    else:
        build(args.root, args.out, args.compact)


if __name__ == "__main__":
    main()
