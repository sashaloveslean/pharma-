#!/usr/bin/env python3
"""Reproduce open-data collection, normalization, audits and optional smoke training."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true")
    p.add_argument("--training-max-rows", type=int, default=12000)
    p.add_argument("--extended-data", action="store_true", help="Also collect QC chromatograms, licensed papers and signal benchmarks (~2.2GB)")
    args = p.parse_args()
    commands = [
        ["open_retention.py", "download-all"],
        ["collect_public_data.py", "--literature"],
        ["open_retention.py", "build", "--compact", "--out", "data/processed/public_retention/report"],
        ["build_public_retention.py", "--resolve-names"],
        ["audit_public_retention.py"],
    ]
    if args.extended_data:
        commands.extend([["collect_chromatograms.py"], ["collect_open_methods.py"],
                         ["collect_signal_benchmarks.py"], ["extract_chromatogram_peaks.py"],
                         ["import_reported_suitability.py"], ["curate_target_evidence.py"],
                         ["report_extended_data.py"]])
    if args.train:
        for name in ["report", "metlin_smrt", "uliege"]:
            commands.append(["train_open_retention.py", "--data",
                             f"data/processed/public_retention/{name}/observations.jsonl", "--out",
                             f"models/public_retention/{name}", "--max-rows", str(args.training_max_rows)])
    for script, *arguments in commands:
        print("Running", script, *arguments, flush=True)
        subprocess.run([sys.executable, "scripts/" + script, *arguments], check=True)


if __name__ == "__main__":
    main()
