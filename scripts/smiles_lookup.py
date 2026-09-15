#!/usr/bin/env python3
"""Resolve molecule names to SMILES through PubChem."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/ConnectivitySMILES,SMILES,IUPACName/JSON"
)


def fetch_pubchem_smiles(name: str, timeout: int = 20) -> dict[str, str]:
    url = PUBCHEM_URL.format(name=quote(name.strip()))
    try:
        with urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(f"PubChem did not find a compound named {name!r}") from exc
        raise RuntimeError(f"PubChem request failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"PubChem request failed: {exc.reason}") from exc

    properties = payload.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        raise ValueError(f"PubChem returned no properties for {name!r}")

    row = properties[0]
    return {
        "name": name,
        "cid": str(row.get("CID", "")),
        "canonical_smiles": row.get("ConnectivitySMILES") or row.get("SMILES", ""),
        "isomeric_smiles": row.get("SMILES", ""),
        "iupac_name": row.get("IUPACName", ""),
    }


def validate_smiles(smiles: str) -> str:
    """Return an RDKit-canonical SMILES or raise for an invalid structure."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for batch SMILES validation") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"PubChem returned an invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def fill_molecule_map(path: Path, apply: bool = False) -> tuple[int, int, int]:
    """Resolve empty SMILES for rows that already have an unambiguous INN."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "inn" not in fieldnames or "smiles" not in fieldnames:
        raise ValueError(f"{path} must contain inn and smiles columns")

    # Tolerate legacy rows whose free-text note contained an unquoted comma.
    for row in rows:
        overflow = row.pop(None, None)
        if overflow:
            note_parts = [(row.get("note") or "").strip(), *[part.strip() for part in overflow]]
            row["note"] = ", ".join(part for part in note_parts if part)

    resolved = failed = skipped = 0
    for row in rows:
        if (row.get("smiles") or "").strip():
            continue
        inn = (row.get("inn") or "").strip()
        note = (row.get("note") or "").strip()
        if not inn or "+" in inn or "EXCLUDE:" in note.upper():
            skipped += 1
            continue
        try:
            result = fetch_pubchem_smiles(inn)
            smiles = validate_smiles(result["isomeric_smiles"] or result["canonical_smiles"])
        except (RuntimeError, ValueError) as exc:
            failed += 1
            print(f"FAILED  {inn}: {exc}")
            continue

        resolved += 1
        print(f"FOUND   {inn}: {smiles} (PubChem CID {result['cid'] or '?'})")
        if apply:
            row["smiles"] = smiles
            row["verified"] = "false"
            provenance = f"auto-filled from PubChem CID {result['cid'] or '?'}; needs review"
            row["note"] = f"{note}; {provenance}" if note else provenance

    if apply:
        # Write atomically so a failed run cannot leave a truncated CSV.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            temporary_path = Path(file.name)
        temporary_path.replace(path)

    return resolved, failed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fill-missing", metavar="CSV")
    parser.add_argument("--apply", action="store_true", help="Write resolved SMILES to the CSV")
    args = parser.parse_args()

    if args.fill_missing:
        resolved, failed, skipped = fill_molecule_map(Path(args.fill_missing), apply=args.apply)
        mode = "updated" if args.apply else "dry run"
        print(f"Summary ({mode}): resolved={resolved}, failed={failed}, skipped={skipped}")
        return
    if not args.name:
        parser.error("provide a compound name or --fill-missing CSV")

    result = fetch_pubchem_smiles(args.name)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Name: {result['name']}")
    print(f"Canonical SMILES: {result['canonical_smiles']}")
    print(f"Isomeric SMILES: {result['isomeric_smiles']}")
    if result["iupac_name"]:
        print(f"IUPAC: {result['iupac_name']}")


if __name__ == "__main__":
    main()
