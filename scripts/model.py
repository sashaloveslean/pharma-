#!/usr/bin/env python3
"""Nearest-analog predictor for HPLC conditions.

With only a few dozen labelled molecules, a trained regressor would overfit.
The robust, chemically meaningful baseline is *nearest-analog transfer*: predict
the conditions of the structurally most similar known molecule(s), weighted by
Tanimoto similarity of Morgan fingerprints. This mirrors how method-development
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

from features import descriptor_vector, morgan_fingerprint, tanimoto


@dataclass
class Analog:
    inn: str
    source_file: str
    similarity: float
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

    def _neighbors(self, fingerprint: list[int], exclude_source: str | None = None) -> list[Analog]:
        analogs: list[Analog] = []
        for row in self.reference:
            if exclude_source and row["source_file"] == exclude_source:
                continue
            similarity = tanimoto(fingerprint, row["fingerprint"])
            analogs.append(
                Analog(
                    inn=row.get("inn", ""),
                    source_file=row["source_file"],
                    similarity=similarity,
                    targets=row["targets"],
                )
            )
        analogs.sort(key=lambda a: a.similarity, reverse=True)
        return analogs[: self.k]

    def predict_from_fingerprint(
        self, fingerprint: list[int], exclude_source: str | None = None
    ) -> dict:
        """Transfer the closest analog's real, coherent method.

        The prediction is one actual method that appeared in an НД (not a blend of
        several, which would give non-physical values and lose the mobile-phase
        recipe). Any field the nearest analog is missing is back-filled from the
        next-closest analogs so the returned method is complete.
        """
        neighbors = self._neighbors(fingerprint, exclude_source=exclude_source)
        all_fields = self.numeric_targets + self.categorical_targets + self.extra_targets

        prediction: dict = {}
        field_source: dict = {}
        for field in all_fields:
            for position, analog in enumerate(neighbors):  # similarity-sorted
                # the closest analog always donates what it has; further analogs
                # only back-fill missing fields if they are similar enough, so a
                # molecule-specific recipe is never copied from a distant match
                if position > 0 and analog.similarity < self.min_backfill_sim:
                    break
                value = analog.targets.get(field)
                if value not in (None, "", []):
                    prediction[field] = value
                    field_source[field] = analog.inn or analog.source_file
                    break
            else:
                prediction[field] = None
                field_source[field] = None

        return {
            "prediction": prediction,
            "field_source": field_source,
            "analogs": [
                {
                    "inn": a.inn,
                    "source_file": a.source_file,
                    "similarity": round(a.similarity, 3),
                }
                for a in neighbors
            ],
        }

    def predict_smiles(self, smiles: str) -> dict:
        fingerprint = morgan_fingerprint(smiles)
        return self.predict_from_fingerprint(fingerprint)

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
