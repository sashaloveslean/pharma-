#!/usr/bin/env python3
"""Inventory all U Liege signals and fetch complete QC series with SHA256 checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from collect_public_data import fetch


def main(args):
    inventory, selected = [], []
    for path in sorted(args.root.glob("uliege_*/source_metadata.json")):
        source = json.loads(path.read_text())
        license_info = source["data_licence"]
        if license_info["short_name"] != "CC BY 4.0":
            raise ValueError(f"Unexpected license for {path}")
        for f in source["files"]:
            if not f["filename"].endswith(".arw"):
                continue
            filename = f["filename"]
            if Path(filename).name != filename:
                raise ValueError("Unsafe file name")
            record = {"dataset": path.parent.name, "version": source["version"],
                      "file_id": f["id"], "filename": filename,
                      "bytes": f["size"], "download_url": f["content_details"]["download_url"],
                      "sha256": f["content_details"]["sha256_hash"], "license": license_info,
                      "is_qc": "-QC-" in filename, "is_blank": "-blank-" in filename,
                      "detector": "UVDAD" if "-UVDAD." in filename else "ELSD",
                      "path": str(path.parent / "signals" / filename)}
            inventory.append(record)
            if args.scope == "all" or (record["is_qc"] and record["detector"] == "UVDAD"):
                selected.append(record)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "chromatogram_inventory.json").write_text(json.dumps(inventory, indent=2))
    manifest_path = args.root / "chromatogram_downloads.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    print(f"Inventory {len(inventory)} signals; selected {len(selected)} / {sum(r['bytes'] for r in selected)/1e9:.2f} GB", flush=True)
    def download(r):
        result = fetch(r["download_url"], Path(r["path"]), r["sha256"])
        return {**r, "download": result, "status": "verified"}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, r): r for r in selected}
        for i, future in enumerate(as_completed(futures), 1):
            r = futures[future]
            key = r["dataset"] + ":" + r["file_id"]
            try:
                previous["files"][key] = future.result()
            except Exception as exc:
                previous["files"][key] = {**r, "status": "failed", "error": str(exc)}
                failures.append(key)
            if i % 10 == 0 or i == len(selected):
                temporary = manifest_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(previous, indent=2))
                temporary.replace(manifest_path)
                print(f"Verified {i}/{len(selected)}, failures {len(failures)}", flush=True)
    if failures:
        raise SystemExit(f"Failed {len(failures)} downloads; rerun resumes verified files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/external"))
    p.add_argument("--scope", choices=["qc", "all"], default="qc",
                   help="qc: all 306 QC DAD signals (~1.9GB); all: all raw signals (~13.5GB)")
    main(p.parse_args())
