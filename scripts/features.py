#!/usr/bin/env python3
"""Molecular featurization for the chromatography-conditions predictor.

Given a SMILES string, produce:
  * a vector of physicochemical descriptors (interpretable, drive retention)
  * a Morgan fingerprint (structural similarity, used for nearest-analog search)

These are the model inputs (X). The chromatography conditions are the targets (y).
"""

from __future__ import annotations

from functools import lru_cache
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
    "CarboxylicAcidCount",
    "AmineCount",
    "AmideCount",
    "PhenolCount",
    "SulfonamideCount",
    "PhosphateCount",
    "HeteroAromaticCount",
    "HalogenCount",
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
    "ChEMBLAcidicPka",
    "ChEMBLBasicPka",
)

FUNCTIONAL_GROUP_SMARTS: dict[str, str] = {
    "CarboxylicAcidCount": "C(=O)[O;H1,-1]",
    "AmineCount": "[NX3;H2,H1,H0;!$(NC=O);!$(NS=O);!$(N[a])]",
    "AmideCount": "[NX3][CX3](=[OX1])",
    "PhenolCount": "[OX2H][c]",
    "SulfonamideCount": "[NX3][SX4](=[OX1])(=[OX1])",
    "PhosphateCount": "[PX4](=[OX1])([OX2])[OX2]",
    "HeteroAromaticCount": "[n,o,s;R]",
    "HalogenCount": "[F,Cl,Br,I]",
}

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
    """Return a normalized parent structure for stable analog comparison.

    RDKit cleanup, largest-fragment selection, neutralization and canonical
    tautomerization prevent salts and alternative tautomer drawings from
    dominating fingerprints. pH-specific charge is modelled separately.
    """
    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = rdMolStandardize.Cleanup(mol)
    mol = rdMolStandardize.FragmentParent(mol, skipStandardize=True)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    mol = rdMolStandardize.CanonicalTautomer(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


PH_PROFILE_POINTS: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0)


@lru_cache(maxsize=4096)
def ph_charge_profile(smiles: str) -> dict[str, list[int]]:
    """Enumerate plausible formal-charge states at selected pH values.

    Dimorphite-DL returns possible protonation states, not quantitative species
    fractions. Each pH is therefore represented by a set of possible net formal
    charges. Failure is explicit so callers can fall back to motif-based logic.
    """
    _require_rdkit()
    from rdkit import Chem, rdBase
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Dimorphite-DL is required for pH-aware ionization: pip install dimorphite-dl"
        ) from exc

    parent = normalize_smiles(smiles)
    profile: dict[str, list[int]] = {}
    for ph in PH_PROFILE_POINTS:
        # Some discarded variants can be chemically invalid; Dimorphite-DL's
        # validated output remains usable, so silence RDKit diagnostics here.
        with rdBase.BlockLogs():
            variants = protonate_smiles(
                parent, ph_min=ph, ph_max=ph, precision=0.5, max_variants=32
            )
            charges = {
                int(Chem.GetFormalCharge(mol))
                for variant in variants
                if (mol := Chem.MolFromSmiles(variant)) is not None
            }
        if not charges:
            mol = Chem.MolFromSmiles(parent)
            charges = {int(Chem.GetFormalCharge(mol))} if mol is not None else set()
        profile[f"{ph:g}"] = sorted(charges)
    return profile


def compute_descriptors(smiles: str) -> dict[str, float]:
    _require_rdkit()
    from rdkit.Chem import Descriptors

    mol = mol_from_smiles(smiles)
    functional_groups = functional_group_counts(mol=mol)
    values: dict[str, float] = {}
    for name in DESCRIPTOR_NAMES:
        if name.startswith(("PubChem", "ChEMBL")):
            values[name] = None
        elif name in functional_groups:
            values[name] = functional_groups[name]
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
        "ChEMBLAcidicPka": (chembl, "cx_most_apka"),
        "ChEMBLBasicPka": (chembl, "cx_most_bpka"),
    }
    for descriptor_name, (source, source_name) in mapping.items():
        value = source.get(source_name)
        try:
            merged[descriptor_name] = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            merged[descriptor_name] = None
    return merged


def functional_group_counts(smiles: str | None = None, mol=None) -> dict[str, float]:
    """Count explicit chromatography-relevant functional groups with SMARTS."""
    _require_rdkit()
    from rdkit import Chem

    if mol is None:
        if not smiles:
            raise ValueError("Either smiles or mol is required")
        mol = mol_from_smiles(smiles)
    return {
        name: float(len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))))
        for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
    }


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
