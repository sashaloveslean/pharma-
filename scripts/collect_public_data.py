#!/usr/bin/env python3
"""Download open METLIN and U Liege data with provenance, licenses and checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

from open_retention import api, table


def fetch(url, path, sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or (sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256):
        temporary = path.with_suffix(path.suffix + ".part")
        # A fresh curl invocation recalculates the resume offset after a timeout.
        # Large public archives must not restart from zero on every retry.
        result = None
        for _ in range(3):
            result = subprocess.run(["curl", "-sSfL", "--connect-timeout", "15", "--max-time", "600",
                                     "--speed-limit", "1024", "--speed-time", "90", "-C", "-",
                                     url, "-o", str(temporary)])
            if result.returncode == 0:
                break
            if result.returncode == 33:  # Server does not support Range requests.
                temporary.unlink(missing_ok=True)
        result.check_returncode()
        if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"SHA256 mismatch: {path}")
        temporary.replace(path)
    return {"path": str(path), "url": url, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/external"))
    p.add_argument("--include-signals", action="store_true", help="Also download multi-GB raw instrument signals")
    p.add_argument("--literature", action="store_true", help="Collect bibliographic records for tenofovir/emtricitabine")
    args = p.parse_args()
    catalog_path = args.root / "public_sources.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    sources = [("metlin_smrt", "https://api.figshare.com/v2/articles/8038913")] + [
        ("uliege_" + sid, "https://data.mendeley.com/public-api/datasets/" + sid)
        for sid in ["csm5gsmr5t", "2w64h8pvkc", "7v5p4gsh4z"]]
    for name, url in sources:
        folder = args.root / name
        folder.mkdir(parents=True, exist_ok=True)
        try:
            metadata_path = folder / "source_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            else:
                metadata = api(url)
                metadata_path.write_text(json.dumps(metadata, indent=2))
            license_info = metadata.get("license") or metadata.get("data_licence")
            if not license_info:
                raise ValueError("Source has no explicit license; download paused for this source")
            record = {"metadata_url": url, "license": license_info, "files": [],
                      "version": metadata.get("version"), "status": "downloading"}
            catalog[name] = record
            for f in metadata["files"]:
                filename = f.get("name") or f.get("filename")
                if not args.include_signals and not filename.lower().endswith((".csv", ".xlsx", ".txt", ".pdf")):
                    continue
                # SMRT descriptors/fingerprints are reproducible locally; avoid redundant 600 MB.
                if name == "metlin_smrt" and filename != "SMRT_dataset.csv":
                    continue
                if Path(filename).name != filename:
                    raise ValueError("Unsafe remote filename")
                details = f.get("content_details", f)
                result = fetch(details["download_url"], folder / filename, details.get("sha256_hash"))
                if f.get("computed_md5") and hashlib.md5((folder / filename).read_bytes()).hexdigest() != f["computed_md5"]:
                    raise ValueError(f"MD5 mismatch: {filename}")
                record["files"].append(result)
                print(name, filename, result["bytes"], flush=True)
            record["status"] = "downloaded" if record["files"] else "metadata_only_raw_signals_available"
            record["excluded_raw_signal_files"] = not args.include_signals
        except Exception as exc:
            catalog[name] = {**catalog.get(name, {}), "status": "failed", "error": str(exc)}
            print(f"{name}: {exc}", flush=True)
        catalog_path.write_text(json.dumps(catalog, indent=2))
    if args.literature:
        folder = args.root / "europepmc"
        folder.mkdir(parents=True, exist_ok=True)
        query = '(tenofovir AND emtricitabine AND (HPLC OR chromatography))'
        cursor, records = "*", []
        while True:
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({
                "query": query, "format": "json", "pageSize": 100, "cursorMark": cursor})
            payload = api(url)
            batch = payload.get("resultList", {}).get("result", [])
            records.extend(batch)
            next_cursor = payload.get("nextCursorMark")
            if not batch or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        path = folder / "tenofovir_emtricitabine_bibliography.json"
        path.write_text(json.dumps({"query": query, "hit_count": payload.get("hitCount"),
                                    "records": records}, indent=2))
        catalog["europepmc"] = {"status": "bibliography_only", "records": len(records),
                                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "note": "Not training labels; full text requires a per-article license check"}
        catalog_path.write_text(json.dumps(catalog, indent=2))
        print("Europe PMC bibliography", len(records), flush=True)
    predret_systems = []
    for path in (args.root / "report").glob("processed_data/*/*_info.tsv"):
        records = table(path)
        if records and records[0].get("source") == "PredRet":
            predret_systems.append(path.parent.name)
    catalog["predret"] = {
        "status": "covered_via_report", "system_ids": predret_systems,
        "direct_export_status": "Website currently requests login; no anonymous export assumed",
        "source_url": "https://predret.org/retention-time-prediction/",
        "license": "CC-BY-SA-4.0", "note": "Do not count these systems twice"}
    catalog_path.write_text(json.dumps(catalog, indent=2))
    if any(v.get("status") == "failed" for v in catalog.values()):
        raise SystemExit("Some sources failed; details in public_sources.json. Other downloads preserved.")


if __name__ == "__main__":
    main()
