"""Versioned chemistry-aware features; enumerated charges are not species fractions."""
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from features import FUNCTIONAL_GROUP_SMARTS
from open_retention import number
from train_open_retention import vector

VERSION = "chemistry_ph_gradient_v1"
HYDRO_VERSION = "chemistry_ph_hydrodynamics_v2"
PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()}
PATTERNS["PhosphonicAcid"] = Chem.MolFromSmarts("P(=O)([O;H1,-1])[O;H1,-1]")
PATTERNS["SulfonicAcid"] = Chem.MolFromSmarts("S(=O)(=O)[O;H1,-1]")
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


@lru_cache(maxsize=50000)
def chemistry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    groups = [len(mol.GetSubstructMatches(pattern)) for pattern in PATTERNS.values()]
    descriptors = [f(mol) for f in (Descriptors.RingCount, Descriptors.NumAromaticRings,
                                   Descriptors.FractionCSP3, Descriptors.HeavyAtomCount)]
    descriptors += [Chem.GetFormalCharge(mol)]
    return descriptors + groups + list(GENERATOR.GetFingerprint(mol)), groups


class ChargeCache:
    def __init__(self, path):
        self.path = path
        self.records = json.loads(path.read_text()) if path.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records))

    def get(self, smiles, ph):
        if ph is None or not 0 < ph <= 14:
            return [np.nan] * 8
        key = hashlib.sha256(f"{smiles}|{ph:.4f}|precision=0.5|max=16".encode()).hexdigest()
        if key not in self.records:
            from dimorphite_dl import protonate_smiles
            try:
                with rdBase.BlockLogs():
                    variants = protonate_smiles(smiles, ph_min=ph, ph_max=ph, precision=0.5, max_variants=16)
                    charges = sorted({Chem.GetFormalCharge(mol) for v in variants
                                      if (mol := Chem.MolFromSmiles(v)) is not None})
                if not charges:
                    raise ValueError("No valid charge states")
                self.records[key] = {"charges": charges, "variants": len(variants), "status": "enumerated"}
            except Exception as exc:
                self.records[key] = {"charges": [], "status": "failed", "error": str(exc)}
        record = self.records[key]
        c = record["charges"]
        if not c:
            return [np.nan] * 8
        return [min(c), max(c), float(0 in c), float(any(q < 0 for q in c)),
                float(any(q > 0 for q in c)), len(c), record["variants"], float(record["variants"] >= 16)]


def enriched_vector(row, numeric, cache, version=VERSION):
    if version not in {VERSION, HYDRO_VERSION}:
        raise ValueError("Unknown feature schema")
    baseline = vector(row, numeric)
    # Replace the compressed 128-bit fingerprint, preserving six base descriptors.
    molecular, groups = chemistry(row["smiles"])
    meta = row["metadata"]
    ph = number(meta.get("eluent.A.pH"))
    if ph == 0:
        ph = None
    charges = cache.get(row["smiles"], ph)
    interactions = [count * ph if ph is not None else np.nan for count in groups]
    length, diameter, flow = [number(meta.get(k)) for k in ("column.length", "column.id", "column.flowrate")]
    volume = math.pi * (diameter / 2) ** 2 * length / 1000 if length and diameter else np.nan
    physical = [volume, flow / volume if flow and volume > 0 else np.nan]
    gradient = row["gradient_program"]
    times = [number(p.get("t [min]")) for p in gradient]
    solvents = []
    if gradient and all(t is not None for t in times) and all(a < b for a, b in zip(times, times[1:])):
        for solvent in ("h2o", "meoh", "acn"):
            concentrations = []
            for p in gradient:
                terms = []
                for channel in "ABCD":
                    fraction = number(p.get(f"{channel} [%]"))
                    component = number(meta.get(f"eluent.{channel}.{solvent}"))
                    if fraction is None or (fraction > 0 and component is None):
                        terms = []
                        break
                    terms.append(fraction * (component or 0) / 100)
                concentrations.append(sum(terms) if terms else np.nan)
            solvents += list(np.interp([0, 1, 2, 3, 5, 10, 15, 20, 30, 60], times, concentrations))
    else:
        solvents = [np.nan] * 30
    values = baseline[:6] + molecular + baseline[134:-2] + interactions + charges + physical + solvents
    if version == HYDRO_VERSION:
        from method_physics import hydrodynamic_features
        values += hydrodynamic_features(row)
    return values + baseline[-2:]
