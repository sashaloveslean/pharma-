#!/usr/bin/env python3
"""Resolve molecule names to SMILES through PubChem."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,IsomericSMILES,IUPACName/JSON"
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
        "canonical_smiles": row.get("CanonicalSMILES") or row.get("ConnectivitySMILES") or row.get("SMILES", ""),
        "isomeric_smiles": row.get("IsomericSMILES") or row.get("SMILES", ""),
        "iupac_name": row.get("IUPACName", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

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
