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

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def compute_descriptors(smiles: str) -> dict[str, float]:
    _require_rdkit()
    from rdkit.Chem import Descriptors

    mol = mol_from_smiles(smiles)
    values: dict[str, float] = {}
    for name in DESCRIPTOR_NAMES:
        func = getattr(Descriptors, name)
        values[name] = float(func(mol))
    return values


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
