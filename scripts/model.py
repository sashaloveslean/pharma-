#!/usr/bin/env python3
"""Nearest-analog predictor for HPLC conditions.

With only a few dozen labelled molecules, a trained regressor would overfit.
The robust, chemically meaningful baseline is *nearest-analog transfer*: predict
the conditions of one coherent method from the closest known molecule. Similarity
combines Morgan-fingerprint Tanimoto with a light descriptor correction for
chromatographically relevant properties. This mirrors how method-development
chemists actually start a new method ("this compound is like X, reuse X's method").

The class is deliberately model-shaped (fit / predict / save / load) so a learned
estimator can be swapped in later once enough labelled data exists.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from features import (
    compute_descriptors,
    ionization_class,
    merge_external_descriptors,
    morgan_fingerprint,
    murcko_scaffold,
    tanimoto,
)


@dataclass
class Analog:
    inn: str
    source_file: str
    similarity: float
    fingerprint_similarity: float
    descriptor_similarity: float | None
    scaffold_similarity: float | None
    ionization_similarity: float | None
    ionization_class: str | None
    targets: dict


class NearestAnalogPredictor:
    def __init__(
        self,
        numeric_targets: list[str],
        categorical_targets: list[str],
        extra_targets: list[str] | None = None,
        k: int = 3,
        min_backfill_sim: float = 0.35,
    ):
        self.numeric_targets = numeric_targets
        self.categorical_targets = categorical_targets
        self.extra_targets = extra_targets or []
        self.k = k
        self.min_backfill_sim = min_backfill_sim
        self.reference: list[dict] = []

    def fit(self, rows: list[dict]) -> "NearestAnalogPredictor":
        self.reference = rows
        return self

    def _neighbors(
        self,
        fingerprint: list[int],
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
        external_properties: dict | None = None,
        exclude_source: str | None = None,
    ) -> list[Analog]:
        analogs: list[Analog] = []
        if descriptors and external_properties:
            descriptors = merge_external_descriptors(descriptors, external_properties)
        target_profile = molecular_profile(smiles, external_properties) if smiles else {}
        for row in self.reference:
            if exclude_source and row["source_file"] == exclude_source:
                continue
            candidate_properties = row.get("external_properties") or {}
            candidate_profile = (
                molecular_profile(row.get("smiles"), candidate_properties)
                if row.get("smiles")
                else {}
            )
            fingerprint_similarity = tanimoto(fingerprint, row["fingerprint"])
            descriptor_similarity = (
                descriptor_similarity_score(descriptors, row.get("descriptors"))
                if descriptors and row.get("descriptors")
                else None
            )
            scaffold_similarity = scaffold_similarity_score(target_profile, candidate_profile)
            ionization_similarity = ionization_similarity_score(target_profile, candidate_profile)
            similarity = combined_similarity(
                fingerprint_similarity,
                descriptor_similarity,
                scaffold_similarity,
                ionization_similarity,
            )
            similarity *= compatibility_penalty(target_profile, candidate_profile)
            analogs.append(
                Analog(
                    inn=row.get("inn", ""),
                    source_file=row["source_file"],
                    similarity=similarity,
                    fingerprint_similarity=fingerprint_similarity,
                    descriptor_similarity=descriptor_similarity,
                    scaffold_similarity=scaffold_similarity,
                    ionization_similarity=ionization_similarity,
                    ionization_class=candidate_profile.get("ionization_class"),
                    targets=row["targets"],
                )
            )
        analogs.sort(key=lambda a: a.similarity, reverse=True)
        return analogs[: self.k]

    def predict_from_fingerprint(
        self,
        fingerprint: list[int],
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
        external_properties: dict | None = None,
        exclude_source: str | None = None,
    ) -> dict:
        """Transfer the closest analog's real, coherent method.

        The prediction is one actual method that appeared in an НД (not a blend of
        several, which would give non-physical values and lose the mobile-phase
        recipe). Missing fields remain missing and are reported explicitly.
        """
        neighbors = self._neighbors(
            fingerprint,
            descriptors=descriptors,
            smiles=smiles,
            external_properties=external_properties,
            exclude_source=exclude_source,
        )
        all_fields = self.numeric_targets + self.categorical_targets + self.extra_targets
        top = neighbors[0] if neighbors else None

        prediction: dict = {}
        field_source: dict = {}
        field_confidence: dict = {}
        missing_fields: list[str] = []
        for field in all_fields:
            value = top.targets.get(field) if top else None
            prediction[field] = value
            if value in (None, "", []):
                field_source[field] = None
                field_confidence[field] = None
                missing_fields.append(field)
            else:
                field_source[field] = top.inn or top.source_file
                field_confidence[field] = round(top.similarity, 3)

        return {
            "prediction": prediction,
            "field_source": field_source,
            "field_confidence": field_confidence,
            "missing_fields": missing_fields,
            "confidence": confidence_summary(neighbors, missing_fields, all_fields),
            "analogs": [
                {
                    "inn": a.inn,
                    "source_file": a.source_file,
                    "similarity": round(a.similarity, 3),
                    "fingerprint_similarity": round(a.fingerprint_similarity, 3),
                    "descriptor_similarity": (
                        round(a.descriptor_similarity, 3)
                        if a.descriptor_similarity is not None
                        else None
                    ),
                    "scaffold_similarity": (
                        round(a.scaffold_similarity, 3)
                        if a.scaffold_similarity is not None
                        else None
                    ),
                    "ionization_similarity": (
                        round(a.ionization_similarity, 3)
                        if a.ionization_similarity is not None
                        else None
                    ),
                    "ionization_class": a.ionization_class,
                }
                for a in neighbors
            ],
        }

    def predict_smiles(self, smiles: str, external_properties: dict | None = None) -> dict:
        fingerprint = morgan_fingerprint(smiles)
        descriptors = merge_external_descriptors(
            compute_descriptors(smiles),
            external_properties,
        )
        return self.predict_from_fingerprint(
            fingerprint,
            descriptors=descriptors,
            smiles=smiles,
            external_properties=external_properties,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "numeric_targets": self.numeric_targets,
            "categorical_targets": self.categorical_targets,
            "extra_targets": self.extra_targets,
            "k": self.k,
            "min_backfill_sim": self.min_backfill_sim,
            "reference": self.reference,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NearestAnalogPredictor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            numeric_targets=payload["numeric_targets"],
            categorical_targets=payload["categorical_targets"],
            extra_targets=payload.get("extra_targets", []),
            k=payload["k"],
            min_backfill_sim=payload.get("min_backfill_sim", 0.35),
        )
        model.reference = payload["reference"]
        return model


def descriptor_matrix(rows: list[dict], descriptor_names: list[str]):
    """Assemble the descriptor feature matrix — for a future learned estimator."""
    return [[row["descriptors"][name] for name in descriptor_names] for row in rows]


def combined_similarity(
    fingerprint_similarity: float,
    descriptor_similarity: float | None,
    scaffold_similarity: float | None,
    ionization_similarity: float | None,
) -> float:
    components = [
        (0.55, fingerprint_similarity),
        (0.25, descriptor_similarity),
        (0.10, scaffold_similarity),
        (0.10, ionization_similarity),
    ]
    weighted = [(weight, value) for weight, value in components if value is not None]
    weight_sum = sum(weight for weight, _ in weighted)
    if not weight_sum:
        return fingerprint_similarity
    return sum(weight * value for weight, value in weighted) / weight_sum


def descriptor_similarity_score(
    target: dict[str, float],
    candidate: dict[str, float],
) -> float:
    """Bounded similarity for chromatographically relevant descriptors.

    This is intentionally conservative: it is not a trained QSRR model, just a
    light-weight correction so MW/logP/TPSA/H-bonding can influence analog choice
    in addition to the Morgan fingerprint.
    """
    weights = {
        "MolWt": (0.12, 250.0),
        "MolLogP": (0.28, 4.0),
        "TPSA": (0.20, 140.0),
        "NumHDonors": (0.08, 5.0),
        "NumHAcceptors": (0.08, 10.0),
        "NumRotatableBonds": (0.06, 12.0),
        "NumAromaticRings": (0.08, 4.0),
        "HeavyAtomCount": (0.10, 40.0),
        "PubChemMolWt": (0.04, 250.0),
        "PubChemXLogP": (0.10, 4.0),
        "PubChemTPSA": (0.08, 140.0),
        "PubChemHDonors": (0.03, 5.0),
        "PubChemHAcceptors": (0.03, 10.0),
        "PubChemRotatableBonds": (0.02, 12.0),
        "ChEMBLMolWt": (0.04, 250.0),
        "ChEMBLLogP": (0.10, 4.0),
        "ChEMBLTPSA": (0.08, 140.0),
        "ChEMBLHDonors": (0.03, 5.0),
        "ChEMBLHAcceptors": (0.03, 10.0),
        "ChEMBLRotatableBonds": (0.02, 12.0),
    }
    score = 0.0
    weight_sum = 0.0
    for name, (weight, scale) in weights.items():
        if target.get(name) is None or candidate.get(name) is None:
            continue
        delta = abs(float(target[name]) - float(candidate[name]))
        score += weight * max(0.0, 1.0 - delta / scale)
        weight_sum += weight
    return score / weight_sum if weight_sum else 0.0


def molecular_profile(smiles: str | None, external_properties: dict | None = None) -> dict:
    if not smiles:
        return {}
    try:
        profile = {
            "scaffold": murcko_scaffold(smiles),
            "ionization_class": ionization_class(smiles),
        }
        chembl = (external_properties or {}).get("chembl") or {}
        if chembl.get("parent_chembl_id"):
            profile["parent_chembl_id"] = chembl["parent_chembl_id"]
        return profile
    except (ValueError, RuntimeError):
        return {}


def scaffold_similarity_score(target: dict, candidate: dict) -> float | None:
    target_scaffold = target.get("scaffold")
    candidate_scaffold = candidate.get("scaffold")
    if not target_scaffold or not candidate_scaffold:
        return None
    if target_scaffold == candidate_scaffold:
        return 1.0
    target_fp = morgan_fingerprint(target_scaffold)
    candidate_fp = morgan_fingerprint(candidate_scaffold)
    return tanimoto(target_fp, candidate_fp)


def ionization_similarity_score(target: dict, candidate: dict) -> float | None:
    target_class = target.get("ionization_class")
    candidate_class = candidate.get("ionization_class")
    if not target_class or not candidate_class:
        return None
    if target_class == candidate_class:
        return 1.0
    if target.get("parent_chembl_id") and target.get("parent_chembl_id") == candidate.get("parent_chembl_id"):
        return 0.9
    if {target_class, candidate_class} == {"weak_acid", "acid"}:
        return 0.8
    if {target_class, candidate_class} == {"weak_acid", "neutral"}:
        return 0.55
    if {target_class, candidate_class} == {"weak_acid", "base"}:
        return 0.1
    if "amphoteric" in {target_class, candidate_class}:
        return 0.65
    return 0.15


def compatibility_penalty(target: dict, candidate: dict) -> float:
    target_class = target.get("ionization_class")
    candidate_class = candidate.get("ionization_class")
    if not target_class or not candidate_class:
        return 1.0
    if target_class == candidate_class:
        return 1.0
    if target.get("parent_chembl_id") and target.get("parent_chembl_id") == candidate.get("parent_chembl_id"):
        return 0.95
    if {target_class, candidate_class} == {"weak_acid", "acid"}:
        return 0.95
    if {target_class, candidate_class} == {"weak_acid", "neutral"}:
        return 0.85
    if {target_class, candidate_class} == {"weak_acid", "base"}:
        return 0.55
    if "amphoteric" in {target_class, candidate_class}:
        return 0.85
    return 0.65


def confidence_summary(
    neighbors: list[Analog],
    missing_fields: list[str],
    all_fields: list[str],
) -> dict:
    top = neighbors[0] if neighbors else None
    structure = top.similarity if top else 0.0
    database_coverage = 1.0 - (len(missing_fields) / len(all_fields) if all_fields else 1.0)
    neighbor_spread = (
        neighbors[0].similarity - neighbors[min(2, len(neighbors) - 1)].similarity
        if len(neighbors) >= 2
        else 0.0
    )
    method_coherence = 1.0 if top else 0.0
    overall = 0.45 * structure + 0.25 * database_coverage + 0.20 * method_coherence + 0.10 * max(0.0, neighbor_spread)
    if overall >= 0.65:
        uncertainty = "low"
    elif overall >= 0.40:
        uncertainty = "medium"
    else:
        uncertainty = "high"
    return {
        "structure_confidence": round(structure, 3),
        "database_coverage": round(database_coverage, 3),
        "method_coherence": round(method_coherence, 3),
        "neighbor_separation": round(max(0.0, neighbor_spread), 3),
        "overall_confidence": round(overall, 3),
        "model_uncertainty": uncertainty,
    }
