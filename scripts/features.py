#!/usr/bin/env python3
"""Molecular featurization for the chromatography-conditions predictor.

Given a SMILES string, produce:
  * a vector of physicochemical descriptors (interpretable, drive retention)
  * a Morgan fingerprint (structural similarity, used for nearest-analog search)

These are the model inputs (X). The chromatography conditions are the targets (y).
"""

from __future__ import annotations

from typing import Sequence

# Physicochemical descriptors most relevant to reversed-phase HPLC retention:
# lipophilicity (LogP), size (MW, heavy atoms), polarity (TPSA), H-bonding,
# ionizable groups, aromaticity, flexibility.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "NumAromaticRings",
    "RingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "LabuteASA",
    "MolMR",
    "FormalCharge",
    "PubChemMolWt",
    "PubChemXLogP",
    "PubChemTPSA",
    "PubChemHDonors",
    "PubChemHAcceptors",
    "PubChemRotatableBonds",
    "ChEMBLMolWt",
    "ChEMBLLogP",
    "ChEMBLTPSA",
    "ChEMBLHDonors",
    "ChEMBLHAcceptors",
    "ChEMBLRotatableBonds",
)

FINGERPRINT_BITS = 1024
FINGERPRINT_RADIUS = 2


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import Descriptors  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RDKit is required for featurization: pip install rdkit"
        ) from exc


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(normalize_smiles(smiles))
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def normalize_smiles(smiles: str) -> str:
    """Canonicalize a SMILES and keep the largest organic fragment.

    This prevents salts/counterions from dominating fingerprints when a user or
    PubChem returns a multi-fragment representation.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if fragments:
        mol = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def compute_descriptors(smiles: str) -> dict[str, float]:
    _require_rdkit()
    from rdkit.Chem import Descriptors

    mol = mol_from_smiles(smiles)
    values: dict[str, float] = {}
    for name in DESCRIPTOR_NAMES:
        if name.startswith(("PubChem", "ChEMBL")):
            values[name] = None
        elif name == "FormalCharge":
            from rdkit import Chem

            values[name] = float(Chem.GetFormalCharge(mol))
        else:
            func = getattr(Descriptors, name)
            values[name] = float(func(mol))
    return values


def merge_external_descriptors(
    descriptors: dict[str, float | None],
    properties: dict | None,
) -> dict[str, float | None]:
    if not properties:
        return descriptors

    merged = dict(descriptors)
    pubchem = properties.get("pubchem") or {}
    chembl = properties.get("chembl") or {}
    mapping = {
        "PubChemMolWt": (pubchem, "MolecularWeight"),
        "PubChemXLogP": (pubchem, "XLogP"),
        "PubChemTPSA": (pubchem, "TPSA"),
        "PubChemHDonors": (pubchem, "HBondDonorCount"),
        "PubChemHAcceptors": (pubchem, "HBondAcceptorCount"),
        "PubChemRotatableBonds": (pubchem, "RotatableBondCount"),
        "ChEMBLMolWt": (chembl, "full_mwt"),
        "ChEMBLLogP": (chembl, "alogp"),
        "ChEMBLTPSA": (chembl, "psa"),
        "ChEMBLHDonors": (chembl, "hbd"),
        "ChEMBLHAcceptors": (chembl, "hba"),
        "ChEMBLRotatableBonds": (chembl, "rtb"),
    }
    for descriptor_name, (source, source_name) in mapping.items():
        value = source.get(source_name)
        try:
            merged[descriptor_name] = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            merged[descriptor_name] = None
    return merged


def descriptor_vector(smiles: str) -> list[float]:
    descriptors = compute_descriptors(smiles)
    return [descriptors[name] for name in DESCRIPTOR_NAMES]


def morgan_fingerprint(smiles: str) -> list[int]:
    _require_rdkit()
    from rdkit.Chem import rdFingerprintGenerator

    mol = mol_from_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    fingerprint = generator.GetFingerprint(mol)
    return list(fingerprint)


def tanimoto(fp_a: Sequence[int], fp_b: Sequence[int]) -> float:
    """Tanimoto similarity between two equal-length binary fingerprints."""
    intersection = sum(1 for a, b in zip(fp_a, fp_b) if a and b)
    union = sum(1 for a, b in zip(fp_a, fp_b) if a or b)
    return intersection / union if union else 0.0


def murcko_scaffold(smiles: str) -> str:
    _require_rdkit()
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return MurckoScaffold.MurckoScaffoldSmilesFromSmiles(
        normalize_smiles(smiles),
        includeChirality=False,
    )


def ionization_class(smiles: str) -> str:
    """Very small rule-based class for analog filtering.

    This is not a pKa predictor. It only catches common pharmaceutical acid/base
    motifs so acidic NSAIDs are not ranked like basic amines solely by descriptors.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = mol_from_smiles(smiles)
    acid_smarts = [
        "C(=O)[O;H1,-1]",  # carboxylic acid / carboxylate
        "S(=O)(=O)[O;H1,-1]",  # sulfonic acid / sulfonate
        "P(=O)([O;H1,-1])[O;H1,-1]",  # phosphonic/phosphoric acid
    ]
    weak_acid_smarts = [
        "S(=O)(=O)N",  # sulfonamide, often weakly acidic/neutral in RP-HPLC
    ]
    base_smarts = [
        "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N[a]);!$([NX3][c])]",
        "[nH0;+0]",
    ]
    has_acid = any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in acid_smarts)
    has_weak_acid = any(
        mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in weak_acid_smarts
    )
    has_base = any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in base_smarts)
    if has_acid and has_base:
        return "amphoteric"
    if has_acid:
        return "acid"
    if has_weak_acid and has_base:
        return "weak_acid"
    if has_weak_acid:
        return "weak_acid"
    if has_base:
        return "base"
    return "neutral"
