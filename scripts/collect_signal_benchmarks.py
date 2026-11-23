#!/usr/bin/env python3
"""Collect additional licensed chromatogram benchmarks, kept separate from QSRR."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

from collect_public_data import fetch
from open_retention import api

SOURCES = [
    ("zenodo_10082436", "https://zenodo.org/api/records/10082436", "HPLC reaction-monitoring chromatograms and analysis code"),
    ("zenodo_14236878", "https://zenodo.org/api/records/14236878", "HPLC method development for glycerol electrooxidation"),
    ("figshare_22182694", "https://api.figshare.com/v2/articles/22182694", "HPLC-DAD multiway signal decomposition benchmark"),
]


def collect(source, root):
    name, url, purpose = source
    folder = root / "signal_benchmarks" / name
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path = folder / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else api(url)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    license_info = metadata.get("license") or metadata["metadata"]["license"]
    if license_info.get("id") != "cc-by-4.0" and license_info.get("name") != "CC BY 4.0":
        raise ValueError("License not on accepted list")
    record = {"source": name, "metadata_url": url, "license": license_info,
              "purpose": purpose, "status": "downloaded_not_normalized", "files": [],
              "training_ready": False, "scope": "Auxiliary signal-processing benchmark; not TFV/FTC evidence"}
    for f in metadata["files"]:
        filename = f.get("key") or f.get("name")
        if Path(filename).name != filename:
            raise ValueError("Unsafe file name")
        download_url = f.get("download_url") or f["links"]["self"]
        path = folder / filename
        downloaded = fetch(download_url, path)
        checksum = f.get("checksum") or ("md5:" + f["computed_md5"] if f.get("computed_md5") else None)
        if checksum and checksum.startswith("md5:"):
            if hashlib.md5(path.read_bytes()).hexdigest() != checksum.split(":", 1)[1]:
                raise ValueError("MD5 mismatch")
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                downloaded["archive_members"] = [{"name": i.filename, "uncompressed_bytes": i.file_size}
                                                   for i in archive.infolist() if not i.is_dir()]
        record["files"].append(downloaded)
    (folder / "manifest.json").write_text(json.dumps(record, indent=2))
    print(name, sum(f["bytes"] for f in record["files"]), "bytes verified", flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/external"))
    args = p.parse_args()
    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(source, pool.submit(collect, source, args.root)) for source in SOURCES]
        for source, future in futures:
            try:
                records.append(future.result())
            except Exception as exc:
                records.append({"source": source[0], "status": "failed", "error": str(exc)})
    (args.root / "signal_benchmarks" / "catalog.json").write_text(json.dumps(records, indent=2))
    if any(r["status"] == "failed" for r in records):
        raise SystemExit("Some benchmark downloads failed; see catalog")
