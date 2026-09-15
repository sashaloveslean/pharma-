#!/usr/bin/env python3
"""Find active-ingredient candidates in extracted regulatory-document text.

The script is deliberately conservative: it only writes recognized INNs with
strong contextual evidence and marks known mixtures/biological products as
excluded. Every automatic decision keeps a short evidence trail in the note.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata


# Russian document forms (including inflections) -> PubChem-compatible INN.
# Adding an alias is safer than sending arbitrary OCR fragments to PubChem.
INN_PATTERNS: dict[str, tuple[str, ...]] = {
    "etoricoxib": (r"эторикоксиб\w*",),
    "benzydamine": (r"бензидамин\w*",),
    "isotretinoin": (r"изотретиноин\w*",),
    "entecavir": (r"энтекавир\w*",),
    "pipecuronium bromide": (r"пипекурони[яю]\s+бромид\w*",),
    "moxifloxacin": (r"моксифлоксацин\w*",),
    "riociguat": (r"риоцигуат\w*",),
    "miramistin": (r"бензилдиметил(?:\[[^]]+\]|[-а-яё]+)*\s*аммони[яй]\w*", r"мирамистин\w*"),
    "meloxicam": (r"мелоксикам\w*",),
    "cinnarizine": (r"циннаризин\w*",),
    "piracetam": (r"пирацетам\w*",),
    "zoledronic acid": (r"золедронов\w+\s+кислот\w*",),
}

NON_SMALL_MOLECULE_PATTERNS = (
    r"полипептид\w*\s+коры\s+головного\s+мозга",
    r"комплекс\s+пептидных\s+фракций",
)


def normalize_filename(value: str) -> str:
    return unicodedata.normalize("NFC", os.path.basename(value)).casefold()


def load_document_text(path: Path) -> dict[str, str]:
    documents: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            source = normalize_filename(chunk.get("metadata", {}).get("source", ""))
            documents.setdefault(source, []).append(chunk.get("text", ""))
    return {source: "\n".join(parts) for source, parts in documents.items()}


def score_candidate(text: str, patterns: tuple[str, ...]) -> tuple[int, str]:
    best_score, best_evidence = 0, ""
    flat = " ".join(text.split())
    for pattern in patterns:
        for match in re.finditer(pattern, flat, flags=re.IGNORECASE):
            start, end = match.span()
            context = flat[max(0, start - 130):min(len(flat), end + 130)]
            lower = context.casefold()
            score = 5
            if "международн" in lower:
                score += 100
            if "действующ" in lower:
                score += 80
            if "стандартного образца" in lower or "стандартный образец" in lower:
                score += 35
            if "время удерживания" in lower or "пика" in lower:
                score += 25
            if score > best_score:
                best_score = score
                best_evidence = context
    return best_score, best_evidence


def classify(text: str) -> dict:
    compact = " ".join(text.split())
    for pattern in NON_SMALL_MOLECULE_PATTERNS:
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            return {"status": "exclude", "reason": "biological mixture", "evidence": match.group(0)}

    scored = []
    for inn, patterns in INN_PATTERNS.items():
        score, evidence = score_candidate(text, patterns)
        if score:
            scored.append((score, inn, evidence))
    scored.sort(reverse=True)

    strong = [item for item in scored if item[0] >= 30]
    strong_names = {item[1] for item in strong}
    if {"cinnarizine", "piracetam"}.issubset(strong_names):
        return {
            "status": "exclude",
            "reason": "combination product: cinnarizine + piracetam",
            "evidence": strong[0][2],
        }
    if not scored or scored[0][0] < 30:
        return {"status": "unresolved", "candidates": scored[:3]}
    top = scored[0]
    return {"status": "resolved", "inn": top[1], "score": top[0], "evidence": top[2]}


def update_map(map_path: Path, chunks_path: Path, apply: bool) -> dict[str, int]:
    documents = load_document_text(chunks_path)
    with map_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"No CSV header in {map_path}")

    counts = {"resolved": 0, "excluded": 0, "unresolved": 0, "already_mapped": 0}
    for row in rows:
        overflow = row.pop(None, None)
        if overflow:
            row["note"] = ", ".join([(row.get("note") or "").strip(), *overflow]).strip(", ")
        if (row.get("inn") or "").strip() or "EXCLUDE:" in (row.get("note") or "").upper():
            counts["already_mapped"] += 1
            continue

        source = row.get("source_file") or ""
        result = classify(documents.get(normalize_filename(source), ""))
        status = result["status"]
        if status == "resolved":
            counts["resolved"] += 1
            print(f"FOUND    {source}: {result['inn']} (score={result['score']})")
            if apply:
                row["inn"] = result["inn"]
                row["note"] = f"auto-extracted INN; score={result['score']}; needs review"
        elif status == "exclude":
            counts["excluded"] += 1
            print(f"EXCLUDE  {source}: {result['reason']}")
            if apply:
                row["note"] = f"EXCLUDE: {result['reason']}; auto-detected; needs review"
        else:
            counts["unresolved"] += 1
            candidates = ", ".join(f"{inn}:{score}" for score, inn, _ in result.get("candidates", []))
            print(f"REVIEW   {source}: {candidates or 'no reliable candidate'}")

    if apply:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=map_path.parent, delete=False
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            temporary_path = Path(file.name)
        temporary_path.replace(map_path)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule-map", default="data/molecule_map.csv")
    parser.add_argument("--chunks", default="data/processed/dissolution_chunks.jsonl")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    counts = update_map(Path(args.molecule_map), Path(args.chunks), args.apply)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
