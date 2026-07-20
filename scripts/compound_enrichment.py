#!/usr/bin/env python3
"""Fetch external compound properties for predictor enrichment.

The cache is deliberately separate from the modelling dataset. It records where
each value came from, so the predictor can use external properties without
pretending they were extracted from the local НД documents.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


PUBCHEM_PROPERTIES = (
    "MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,"
    "HBondAcceptorCount,RotatableBondCount,FormalCharge,CanonicalSMILES,"
    "IsomericSMILES,InChIKey,IUPACName"
)
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/" + PUBCHEM_PROPERTIES + "/JSON"
)
CHEMBL_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={name}&limit=5"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"


def _read_json_url(url: str, timeout: int = 20) -> dict:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_pubchem_properties(name: str) -> dict:
    url = PUBCHEM_URL.format(name=quote(name.strip()))
    try:
        payload = _read_json_url(url)
    except HTTPError as exc:
        if exc.code == 404:
            return {}
        raise RuntimeError(f"PubChem request failed for {name!r}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"PubChem request failed for {name!r}: {exc.reason}") from exc

    rows = payload.get("PropertyTable", {}).get("Properties", [])
    if not rows:
        return {}
    row = rows[0]
    row["source_name"] = "PubChem PUG REST"
    row["source_type"] = "calculated_or_curated_property"
    row["retrieval_date"] = date.today().isoformat()
    return row


def fetch_chembl_properties(name: str) -> dict:
    try:
        search = _read_json_url(CHEMBL_SEARCH_URL.format(name=quote(name.strip())))
    except HTTPError as exc:
        if exc.code == 404:
            return {}
        raise RuntimeError(f"ChEMBL search failed for {name!r}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"ChEMBL search failed for {name!r}: {exc.reason}") from exc

    molecules = search.get("molecules") or []
    if not molecules:
        return {}

    exact_name = name.strip().lower()
    selected = None
    for molecule in molecules:
        names = {
            str(molecule.get("pref_name") or "").lower(),
            str(molecule.get("molecule_synonyms") or "").lower(),
        }
        if exact_name in names:
            selected = molecule
            break
    if selected is None:
        selected = molecules[0]

    chembl_id = selected.get("molecule_chembl_id")
    if not chembl_id:
        return {}

    try:
        molecule = _read_json_url(CHEMBL_MOLECULE_URL.format(chembl_id=quote(chembl_id)))
    except (HTTPError, URLError):
        molecule = selected

    props = molecule.get("molecule_properties") or {}
    structures = molecule.get("molecule_structures") or {}
    hierarchy = molecule.get("molecule_hierarchy") or {}
    result = {
        "molecule_chembl_id": molecule.get("molecule_chembl_id") or chembl_id,
        "parent_chembl_id": hierarchy.get("parent_chembl_id"),
        "pref_name": molecule.get("pref_name"),
        "canonical_smiles": structures.get("canonical_smiles"),
        "standard_inchi_key": structures.get("standard_inchi_key"),
        "full_mwt": props.get("full_mwt"),
        "alogp": props.get("alogp"),
        "psa": props.get("psa"),
        "hba": props.get("hba"),
        "hbd": props.get("hbd"),
        "rtb": props.get("rtb"),
        "source_name": "ChEMBL Web Services",
        "source_type": "curated_druglike_property",
        "retrieval_date": date.today().isoformat(),
    }
    return {key: value for key, value in result.items() if value not in (None, "")}


def enrich_name(name: str) -> dict:
    return {
        "source_value": name,
        "retrieval_date": date.today().isoformat(),
        "pubchem": fetch_pubchem_properties(name),
        "chembl": fetch_chembl_properties(name),
    }


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--out", default="data/processed/compound_properties.json")
    parser.add_argument("--name", action="append", help="Fetch one or more molecule names")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    names: list[str] = []
    if args.name:
        names.extend(args.name)
    else:
        with open(args.molecule_map, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                inn = (row.get("inn") or "").strip()
                smiles = (row.get("smiles") or "").strip()
                if inn and smiles and "allergen" not in inn.lower():
                    names.append(inn)

    cache_path = Path(args.out)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {} if args.refresh else load_cache(cache_path)

    fetched = 0
    for name in sorted(set(names), key=str.lower):
        key = name.strip().lower()
        if not key or (key in cache and not args.refresh):
            continue
        try:
            cache[key] = enrich_name(name)
            fetched += 1
            print(f"enriched {name}")
        except RuntimeError as exc:
            cache[key] = {"source_value": name, "error": str(exc), "retrieval_date": date.today().isoformat()}
            print(f"skip {name}: {exc}")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(cache)} compound records ({fetched} fetched) -> {cache_path}")


if __name__ == "__main__":
    main()
