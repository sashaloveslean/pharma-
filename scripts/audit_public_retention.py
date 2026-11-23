#!/usr/bin/env python3
"""Summarize source coverage and flag cross-source duplicate candidates."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("data/processed/public_retention"))
    args = p.parse_args()
    summary, identities, measurements = {}, {}, {}
    for path in sorted(args.root.glob("*/observations.jsonl")):
        identities[path.parent.name], measurements[path.parent.name] = set(), set()
        systems, rows = Counter(), 0
        for line in path.open():
            r = json.loads(line)
            rows += 1
            systems[r["system_id"]] += 1
            identities[path.parent.name].add(r["compound_id"])
            measurements[path.parent.name].add((r["compound_id"], round(r["retention_time_min"], 5)))
        summary[path.parent.name] = {"rows": rows, "structures": len(identities[path.parent.name]),
                                     "systems": len(systems)}
    overlap = {}
    for a, b in combinations(identities, 2):
        overlap[f"{a} / {b}"] = {
            "shared_structures": len(identities[a] & identities[b]),
            "same_structure_and_rt_candidates": len(measurements[a] & measurements[b]),
            "action": "Keep sources separate; reconcile method provenance before pooling"}
    report = {"datasets": summary, "cross_source_overlap": overlap,
              "unique_structures_union": len(set().union(*identities.values())) if identities else 0,
              "warning": "Summed rows are not a deduplicated training set; matching RT alone is not proof of duplication"}
    (args.root / "corpus_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
