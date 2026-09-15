#!/usr/bin/env python3
"""Build auditable retention observations and a strict QSRR training table.

The extractor deliberately separates automatically discovered observations from
training-ready rows. A row enters the latter only when tR, t0 and critical HPLC
conditions are present; log(k) is never inferred from run time or column volume.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


RETENTION_CUE_RE = re.compile(r"врем\w*\s+удерживан\w*|retention\s+time|\bRT\b", re.I)
MINUTES_RE = re.compile(
    r"(?P<first>\d{1,3}(?:[.,]\d+)?)\s*(?:[-–—]\s*(?P<second>\d{1,3}(?:[.,]\d+)?))?\s*мин",
    re.I,
)
T0_RE = re.compile(
    r"(?:мертв\w*\s+врем\w*|dead\s+time|void\s+time|t\s*[₀0])\D{0,30}"
    r"(\d{1,3}(?:[.,]\d+)?)\s*мин",
    re.I,
)
T0_REVERSE_RE = re.compile(
    r"(?:пик\w*\s+.{0,50})?(?:врем\w*\s+удерживан\w*)\D{0,20}"
    r"(\d{1,3}(?:[.,]\d+)?)\s*мин\D{0,100}"
    r"(?:использ\w*|принима\w*|определ\w*)\D{0,40}"
    r"(?:как\s+)?(?:t\s*[₀0o]|rt\s*[₀0o]|мертв\w*\s+врем\w*)",
    re.I,
)

FIELDS = [
    "observation_id", "review_status", "inn", "smiles", "column_chemistry",
    "column_length_mm", "column_id_mm", "particle_um", "modifier",
    "organic_percent", "pH", "buffer", "flow_ml_min", "temperature_c",
    "t0_min", "t0_source", "t0_porosity_assumption", "retention_time_min", "retention_time_low_min",
    "retention_time_high_min", "log_k", "source_file", "source_page",
    "source_chunk_id", "condition_page_distance", "extraction_method",
    "extraction_confidence", "validation_status",
    "validation_issues", "evidence",
]


def number(value: str) -> float:
    return float(value.replace(",", "."))


def truthy(value) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "да"}


def load_molecule_map(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as file:
        return {row["source_file"]: row for row in csv.DictReader(file)}


def load_conditions(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            grouped.setdefault(row["source"], []).append(row)
    return grouped


def condition_score(row: dict) -> int:
    fields = (
        "column_phase", "column_length_mm", "column_id_mm", "particle_um",
        "mobile_phase_components", "mobile_phase_ratio", "flow_ml_min",
        "mobile_phase_ph", "column_temp_c",
    )
    return sum(row.get(field) not in (None, "", []) for field in fields)


def nearest_conditions(rows: list[dict], page: int | None, max_page_distance: int = 3) -> dict:
    if not rows:
        return {}
    ordered = sorted(
        rows,
        key=lambda row: (
            abs((row.get("start_page") or page or 0) - (page or 0)),
            -condition_score(row),
        ),
    )
    nearest = ordered[0]
    merged = dict(nearest)
    used_distances = [abs((nearest.get("start_page") or page or 0) - (page or 0))]
    merge_fields = (
        "column_phase", "column_length_mm", "column_id_mm", "particle_um",
        "mobile_phase_components", "mobile_phase_ratio", "mobile_phase_ph",
        "flow_ml_min", "column_temp_c", "solvents", "has_buffer",
    )
    for candidate in ordered[1:]:
        distance = abs((candidate.get("start_page") or page or 0) - (page or 0))
        if distance > max_page_distance:
            continue
        used = False
        for field in merge_fields:
            if field == "solvents" and candidate.get(field):
                combined = list(dict.fromkeys([*(merged.get(field) or []), *candidate[field]]))
                if combined != (merged.get(field) or []):
                    merged[field] = combined
                    used = True
                continue
            if merged.get(field) in (None, "", []) and candidate.get(field) not in (None, "", []):
                merged[field] = candidate[field]
                used = True
        if used:
            used_distances.append(distance)
    merged["condition_page_distance"] = max(used_distances)
    return merged


def mobile_phase_features(condition: dict) -> tuple[str | None, float | None]:
    components = condition.get("mobile_phase_components") or []
    total = sum(float(item.get("part", 0)) for item in components)
    for item in components:
        solution = str(item.get("solution", "")).lower()
        if "ацетонитрил" in solution or "acetonitrile" in solution:
            return "acetonitrile", 100 * float(item["part"]) / total if total else None
        if "метанол" in solution or "methanol" in solution:
            return "methanol", 100 * float(item["part"]) / total if total else None
    solvent_names = set(condition.get("solvents") or [])
    organics = [name for name in ("acetonitrile", "methanol") if name in solvent_names]
    ratio = condition.get("mobile_phase_ratio")
    if len(organics) == 1 and ratio:
        try:
            parts = [number(value) for value in str(ratio).split(":")]
        except ValueError:
            parts = []
        # Without resolved component ordering, the lower fraction is the safest
        # RP-HPLC provisional assumption and remains subject to manual review.
        if len(parts) == 2 and sum(parts) > 0:
            return organics[0], 100 * min(parts) / sum(parts)
    if len(organics) == 1:
        return organics[0], None
    return None, None


def retention_candidates(text: str, inn: str) -> list[tuple[float | None, float, float, str, str]]:
    """Return exact/range tR candidates for human review.

    INNs in the map are often English while source PDFs are Russian, so an INN
    text match must not suppress discovery. Provenance and review remain strict.
    """
    found = []
    for cue in RETENTION_CUE_RE.finditer(text):
        start = max(0, cue.start() - 180)
        end = min(len(text), cue.end() + 260)
        window = text[start:end]
        match = MINUTES_RE.search(text, cue.end(), min(len(text), cue.end() + 240))
        if not match:
            continue
        low = number(match.group("first"))
        high = number(match.group("second")) if match.group("second") else low
        if not (0.05 <= low <= high <= 300):
            continue
        exact = low if high == low else None
        between = text[cue.end():match.start()]
        strong_context = bool(re.search(r"предел|состав|около|типич|ориентировоч|равн", between, re.I))
        confidence = "high" if strong_context and len(between) <= 180 else (
            "medium" if len(between) <= 100 else "low"
        )
        evidence = re.sub(r"\s+", " ", text[start:min(len(text), match.end() + 80)]).strip()
        found.append((exact, low, high, evidence, confidence))
    return found


def compute_log_k(retention_time: float | None, t0: float | None) -> float | None:
    if retention_time is None or t0 is None or t0 <= 0 or retention_time <= t0:
        return None
    return math.log10((retention_time - t0) / t0)


def extract_t0(text: str) -> float | None:
    """Extract t0 whether the label appears before or after the numeric time."""
    match = T0_RE.search(text) or T0_REVERSE_RE.search(text)
    return number(match.group(1)) if match else None


def estimate_t0(condition: dict, porosity: float) -> float | None:
    """Estimate column dead time in minutes from interstitial volume and flow."""
    try:
        length = float(condition["column_length_mm"])
        diameter = float(condition["column_id_mm"])
        flow = float(condition["flow_ml_min"])
    except (KeyError, TypeError, ValueError):
        return None
    if length <= 0 or diameter <= 0 or flow <= 0:
        return None
    geometric_volume_ml = math.pi * (diameter / 2) ** 2 * length / 1000
    return geometric_volume_ml * porosity / flow


def validate(row: dict) -> list[str]:
    required = {
        "smiles": "missing_smiles",
        "column_chemistry": "missing_column",
        "modifier": "missing_modifier",
        "organic_percent": "missing_organic_percent",
        "flow_ml_min": "missing_flow",
        "retention_time_min": "missing_exact_tR",
        "t0_min": "missing_t0",
    }
    issues = [issue for field, issue in required.items() if row.get(field) in (None, "")]
    if truthy(row.get("buffer")) and row.get("pH") in (None, ""):
        issues.append("missing_buffer_pH")
    if row.get("t0_source") == "estimated_column_geometry":
        issues.append("estimated_t0_not_measured")
    if row.get("retention_time_min") and row.get("t0_min"):
        if float(row["retention_time_min"]) <= float(row["t0_min"]):
            issues.append("tR_not_greater_than_t0")
    return issues


def discover_observations(
    chunks_path: Path,
    molecule_map: dict[str, dict],
    conditions: dict[str, list[dict]],
    porosity: float = 0.68,
) -> list[dict]:
    observations: list[dict] = []
    seen: set[tuple] = set()
    with chunks_path.open(encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            source = Path(chunk["metadata"]["source"]).name
            mapping = molecule_map.get(source) or {}
            inn, smiles = (mapping.get("inn") or "").strip(), (mapping.get("smiles") or "").strip()
            if not inn or not smiles or "+" in inn:
                continue
            page = chunk["metadata"].get("start_page")
            condition = nearest_conditions(conditions.get(source, []), page)
            modifier, organic_percent = mobile_phase_features(condition)
            t0 = extract_t0(chunk["text"])
            estimated_t0 = estimate_t0(condition, porosity)
            resolved_t0 = t0 if t0 is not None else estimated_t0
            t0_source = "measured_document" if t0 is not None else (
                "estimated_column_geometry" if estimated_t0 is not None else None
            )
            for exact, low, high, evidence, confidence in retention_candidates(chunk["text"], inn):
                # A marker peak described as t0 is not an analyte retention observation.
                if t0 is not None and exact is not None and math.isclose(exact, t0, rel_tol=0, abs_tol=1e-9):
                    continue
                identity = (source, page, low, high)
                if identity in seen:
                    continue
                seen.add(identity)
                row = {
                    "observation_id": f"auto-{len(observations) + 1:05d}",
                    "review_status": "needs_review",
                    "inn": inn,
                    "smiles": smiles,
                    "column_chemistry": condition.get("column_phase"),
                    "column_length_mm": condition.get("column_length_mm"),
                    "column_id_mm": condition.get("column_id_mm"),
                    "particle_um": condition.get("particle_um"),
                    "modifier": modifier,
                    "organic_percent": round(organic_percent, 3) if organic_percent is not None else None,
                    "pH": condition.get("mobile_phase_ph"),
                    "buffer": bool(condition.get("has_buffer") or set(condition.get("solvents") or []) & {"phosphate_buffer", "acetate_buffer"}),
                    "flow_ml_min": condition.get("flow_ml_min"),
                    "temperature_c": condition.get("column_temp_c"),
                    "t0_min": round(resolved_t0, 4) if resolved_t0 is not None else None,
                    "t0_source": t0_source,
                    "t0_porosity_assumption": porosity if t0_source == "estimated_column_geometry" else None,
                    "retention_time_min": exact,
                    "retention_time_low_min": low,
                    "retention_time_high_min": high,
                    "log_k": compute_log_k(exact, resolved_t0),
                    "source_file": source,
                    "source_page": page,
                    "source_chunk_id": chunk.get("id"),
                    "condition_page_distance": condition.get("condition_page_distance"),
                    "extraction_method": "regex_v1",
                    "extraction_confidence": confidence,
                    "evidence": evidence,
                }
                issues = validate(row)
                row["validation_status"] = "ready" if not issues else "incomplete"
                row["validation_issues"] = ";".join(issues)
                observations.append(row)
    return observations


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)


def load_curated(path: Path) -> list[dict]:
    """Load the human-owned table; create an empty template without overwriting it."""
    if not path.exists():
        write_csv(path, [])
        return []
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        existing_fields = reader.fieldnames or []
    if not rows and existing_fields != FIELDS:
        write_csv(path, [])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/dissolution_chunks.jsonl")
    parser.add_argument("--conditions", default="data/processed/chromatography_conditions.jsonl")
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--observations-out", default="data/processed/qsrr_observations.csv")
    parser.add_argument("--curated", default="data/qsrr_curated.csv")
    parser.add_argument("--training-out", default="data/processed/qsrr_training.csv")
    parser.add_argument("--provisional-out", default="data/processed/qsrr_provisional.csv")
    parser.add_argument("--report-out", default="data/processed/qsrr_report.json")
    parser.add_argument("--porosity", type=float, default=0.68)
    args = parser.parse_args()

    observations = discover_observations(
        Path(args.chunks),
        load_molecule_map(Path(args.molecule_map)),
        load_conditions(Path(args.conditions)),
        porosity=args.porosity,
    )
    curated = load_curated(Path(args.curated))
    training = []
    for row in curated:
        issues = validate(row)
        row["log_k"] = compute_log_k(
            number(row["retention_time_min"]) if row.get("retention_time_min") else None,
            number(row["t0_min"]) if row.get("t0_min") else None,
        )
        row["validation_status"] = "ready" if not issues else "incomplete"
        row["validation_issues"] = ";".join(issues)
        if not issues and row.get("review_status") == "approved":
            training.append(row)
    write_csv(Path(args.observations_out), observations)
    provisional = [
        row for row in observations
        if row.get("log_k") is not None
        and not set(filter(None, row["validation_issues"].split(";")))
        - {"estimated_t0_not_measured"}
    ]
    write_csv(Path(args.provisional_out), provisional)
    write_csv(Path(args.training_out), training)
    issue_counts: dict[str, int] = {}
    for row in observations:
        for issue in filter(None, row["validation_issues"].split(";")):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    report = {
        "candidate_observations": len(observations),
        "curated_observations": len(curated),
        "provisional_rows": len(provisional),
        "training_ready_and_approved": len(training),
        "unique_molecules": len({row["inn"] for row in observations}),
        "issue_counts": issue_counts,
        "warning": "Do not train QSRR until rows are reviewed and include measured t0 and tR.",
    }
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
