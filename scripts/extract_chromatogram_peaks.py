#!/usr/bin/env python3
"""Derive provisional half-height widths and same-run Rs from real QC signals."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import openpyxl
import pandas as pd
from scipy.signal import find_peaks

ANALYTES = ("uracil", "3-cyanopyridine", "ibuprofen")
FORMULA_SOURCE = "https://help.waters.com/content/dam/waters/pt_br/support/usermanuals/2022/715007659/715007659v01.pdf"


def filename_ph(code):
    # These two codes are used by this dataset in place of decimal points.
    return {"35": 3.5, "65": 6.5}.get(code, float(code))


def read_dad(path):
    # System 3 wraps each whole numeric CSV line in an extra quote pair.
    # Unwrap only that transport layer, keeping the original raw file untouched.
    with path.open(encoding="utf-8-sig") as file:
        lines = []
        for line in file:
            line = line.rstrip("\r\n")
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            lines.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(lines)), skiprows=[1])
    times = frame.iloc[:, 0].to_numpy(float)
    wavelengths = np.asarray([float(c) for c in frame.columns[1:]])
    signals = frame.iloc[:, 1:].to_numpy(float)
    if len(times) < 10 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid/non-increasing time axis")
    return times, wavelengths, signals


def half_height_width(times, corrected, peak):
    height = corrected[peak]
    if height <= 0:
        raise ValueError("Non-positive peak")
    half = height / 2
    left, right = peak, peak
    while left > 0 and corrected[left] > half:
        left -= 1
    while right < len(times)-1 and corrected[right] > half:
        right += 1
    if corrected[left] > half or corrected[right] > half:
        raise ValueError("Half-height crossing outside integration window")
    l = np.interp(half, corrected[[left, left+1]], times[[left, left+1]])
    r = np.interp(half, corrected[[right, right-1]], times[[right, right-1]])
    if r <= l:
        raise ValueError("Invalid width")
    return float(r-l), float(l), float(r)


def measure_peak(times, signal, expected):
    low, high = min(expected), max(expected)
    mask = (times >= low-0.5) & (times <= high+0.5)
    x, raw = times[mask], signal[mask]
    if len(x) < 12 or not np.all(np.isfinite(raw)):
        return {"status": "invalid_signal_window"}
    edges = max(3, min(8, len(x)//8))
    baseline = np.linspace(np.median(raw[:edges]), np.median(raw[-edges:]), len(raw))
    y = raw - baseline
    diff = np.diff(y)
    noise = max(float(np.median(np.abs(diff-np.median(diff))) / (0.67449*np.sqrt(2))), 1e-12)
    peaks, props = find_peaks(y, prominence=6*noise, distance=3)
    keep = [i for i,p in enumerate(peaks) if low-0.15 <= x[p] <= high+0.15]
    if not keep:
        return {"status": "no_peak_in_reference_window", "noise": noise}
    best = max(keep, key=lambda i: props["prominences"][i])
    peak = peaks[best]
    try:
        width, left, right = half_height_width(x, y, peak)
    except ValueError as exc:
        return {"status": "width_failed", "reason": str(exc)}
    sampling = float(np.median(np.diff(x)))
    flags = []
    if width/sampling < 5:
        flags.append("fewer_than_5_sampling_intervals_across_width")
    if len(keep) > 1:
        flags.append("multiple_peak_candidates_in_reference_window")
    snr = float(y[peak]/noise)
    if snr < 10:
        flags.append("low_snr")
    asymmetry = (right-x[peak]) / max(x[peak]-left, 1e-12)
    if not 1/3 <= asymmetry <= 3:
        flags.append("strong_asymmetry_or_integration_problem")
    return {"status": "derived_needs_review", "retention_time_min": float(x[peak]),
            "width_half_height_min": width, "width_definition": "baseline-corrected local half-height",
            "left_half_height_min": left, "right_half_height_min": right,
            "height": float(y[peak]), "signal_to_noise": snr, "sampling_interval_min": sampling,
            "width_sampling_intervals": width/sampling, "quality_flags": flags,
            "reference_rt_min_max": [float(low), float(high)], "reported_width": None,
            "reviewed": False, "training_ready": False}


def rs_half_height(first, second):
    if first["width_half_height_min"] <= 0 or second["width_half_height_min"] <= 0:
        raise ValueError("Non-positive width")
    return 1.18*abs(second["retention_time_min"]-first["retention_time_min"])/(first["width_half_height_min"]+second["width_half_height_min"])


def reference_windows(path):
    sheet = iter(openpyxl.load_workbook(path, read_only=True, data_only=True).active.values)
    headers = next(sheet)
    windows = defaultdict(list)
    for values in sheet:
        row = dict(zip(headers, values))
        if row.get("is_QC") != 1:
            continue
        key = (int(str(row["system"]).split()[-1]), float(row["pH_target"]), float(row["gradient_time"]), row["compound"])
        windows[key].append(float(row["retention_time"]))
    return windows


def qc_author_links(summary, inventory):
    sheet = iter(openpyxl.load_workbook(summary, read_only=True, data_only=True).active.values)
    headers = next(sheet)
    runs = defaultdict(dict)
    for values in sheet:
        r = dict(zip(headers, values))
        if r.get("is_QC") != 1:
            continue
        key = (int(r["system"].split()[-1]), float(r["pH_target"]), float(r["gradient_time"]))
        ident = (r["date_sequence"], r["experiment_id"], r["injection_order"])
        runs[key].setdefault(ident, {})[r["compound"]] = float(r["retention_time"])
    groups = defaultdict(list)
    for r in inventory:
        if not r["is_qc"] or r["detector"] != "UVDAD":
            continue
        match = re.search(r"system(\d+)-pH([\d.]+)-([\d.]+)min-QC", r["filename"])
        if match:
            system, ph, gradient = match.groups()
            groups[(int(system), filename_ph(ph), float(gradient))].append(r)
    links = {}
    for key, raw_files in groups.items():
        author_runs = runs.get(key, {})
        if len(raw_files) != len(author_runs):
            continue  # Never force alignment when injections are missing or extra.
        for raw, (ident, rt) in zip(sorted(raw_files, key=lambda r:r["filename"]), sorted(author_runs.items())):
            if set(rt) != set(ANALYTES):
                continue
            links[raw["file_id"]] = {"sequence_date": ident[0], "experiment_id": ident[1],
                                     "injection_order": ident[2], "reported_rt_min": rt,
                                     "link_rule": "same condition, equal run counts, chronology; requires observed RT check"}
    return links


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    windows = reference_windows(args.summary)
    manifest = json.loads(args.manifest.read_text())
    inventory = json.loads((args.manifest.parent / "chromatogram_inventory.json").read_text())
    author_links = qc_author_links(args.summary, inventory)
    widths, pairs, files = [], [], []
    example = None
    for record in manifest["files"].values():
        if record["status"] != "verified" or not record["is_qc"] or record["detector"] != "UVDAD":
            continue
        path = Path(record["path"])
        match = re.search(r"system(\d+)-pH([\d.]+)-([\d.]+)min-QC", path.name)
        if not match:
            files.append({"file": str(path), "status": "filename_not_mapped"})
            continue
        system_code, ph_code, gradient_code = match.groups()
        system, ph, gradient = int(system_code), filename_ph(ph_code), float(gradient_code)
        key = (system, ph, gradient)
        run_id = record["dataset"] + ":" + record["file_id"]
        try:
            if not all(windows.get(key+(name,)) for name in ANALYTES):
                files.append({"file": str(path), "run_id": run_id, "status": "missing_reference_windows"})
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Signal checksum mismatch")
            times, wavelengths, matrix = read_dad(path)
            options = []
            for wavelength in (220, 254, 260, 280):
                index = int(np.argmin(abs(wavelengths-wavelength)))
                peaks = {name: measure_peak(times, matrix[:, index], windows[key+(name,)]) for name in ANALYTES
                         if windows.get(key+(name,))}
                valid = sum(p["status"] == "derived_needs_review" for p in peaks.values())
                min_snr = min((p.get("signal_to_noise", 0) for p in peaks.values()), default=0)
                options.append(((valid, min_snr), index, peaks))
            _, channel, peaks = max(options, key=lambda option: option[0])
            link = author_links.get(record["file_id"])
            rt_verified = bool(link) and len(peaks) == 3 and all(
                p.get("status") == "derived_needs_review" and
                abs(p["retention_time_min"]-link["reported_rt_min"][name]) <= max(0.04, 2*p["sampling_interval_min"])
                for name, p in peaks.items())
            for name, peak in peaks.items():
                widths.append({"run_id": run_id, "compound": name, "system": int(system),
                               "target_ph": ph, "nominal_gradient_min": gradient,
                               "filename_ph_code": ph_code,
                               "wavelength_nm": float(wavelengths[channel]), "file": str(path),
                               "source_url": record["download_url"], "source_sha256": record["sha256"],
                               "license": record["license"], "identity_status": "putative_QC_composition_and_RT_window",
                               "signal_type": "real_same_injection_QC_mixture", **peak})
                widths[-1].update(author_run_link=link if rt_verified else None,
                                  author_link_status="chronology_and_RT_checked" if rt_verified else "unresolved",
                                  reported_tr_min=link["reported_rt_min"][name] if rt_verified else None)
            for first, second in zip(ANALYTES, ANALYTES[1:]):
                a, b = peaks.get(first, {}), peaks.get(second, {})
                if a.get("status") != "derived_needs_review" or b.get("status") != "derived_needs_review":
                    continue
                pairs.append({"run_id": run_id, "compound_1": first, "compound_2": second,
                              "derived_rs_half_height": rs_half_height(a, b), "reported_rs": None,
                              "width_1_half_height_min": a["width_half_height_min"],
                              "width_2_half_height_min": b["width_half_height_min"],
                              "tr_1_min": a["retention_time_min"], "tr_2_min": b["retention_time_min"],
                              "wavelength_nm": float(wavelengths[channel]), "file": str(path),
                              "source_sha256": record["sha256"], "formula": "1.18*abs(tR2-tR1)/(wh1+wh2)",
                              "formula_source": FORMULA_SOURCE, "quality_flags": sorted(set(a["quality_flags"]+b["quality_flags"])),
                              "status": "derived_needs_review", "reviewed": False, "training_ready": False})
            files.append({"file": str(path), "run_id": run_id, "status": "processed", "peaks": len(peaks)})
            files[-1]["author_run_RT_checked"] = rt_verified
            if example is None and len(peaks) == 3 and all(p["status"] == "derived_needs_review" for p in peaks.values()):
                example = (times, matrix[:, channel], peaks, path.name, float(wavelengths[channel]))
        except Exception as exc:
            files.append({"file": str(path), "run_id": run_id, "status": "failed", "error": str(exc)})
    for name, rows in [("peak_width_candidates", widths), ("resolution_candidates", pairs)]:
        with (args.out / (name + ".jsonl")).open("w") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
    report = {"files": files, "processed_signals": sum(f["status"] == "processed" for f in files),
              "peak_windows": len(widths), "derived_widths": sum(r["status"] == "derived_needs_review" for r in widths),
              "derived_resolution_pairs": len(pairs), "approved_training_rows": 0,
              "signals_linked_to_author_runs_with_rt_check": sum(f.get("author_run_RT_checked", False) for f in files),
              "width_quality_flags": dict(Counter(f for r in widths for f in r.get("quality_flags", []))),
              "limitations": ["Only QC mixture of uracil, 3-cyanopyridine, ibuprofen; no TFV/FTC chromatograms",
                              "QC composition/RT-window identities; only a subset has chronology and author RT checked per injection",
                              "Local linear baseline and automatic integration require visual review",
                              "Estimated half-height Rs, not author-reported Rs or a trained separation model",
                              "QC compounds are widely separated; near-coelution is not represented"]}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    if example:
        plot_example(args.out, example)
    print(json.dumps({k:v for k,v in report.items() if k != "files"}, indent=2))


def plot_example(out, example):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(out / ".cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times, signal, peaks, filename, wavelength = example
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(times, signal, lw=0.8)
    axes[0].set(title=f"Real U Liege QC trace / {wavelength:.1f} nm", ylabel="Absorbance (source units)")
    for name, peak in peaks.items():
        rt = peak["retention_time_min"]
        axes[0].axvline(rt, ls="--", lw=0.6)
        axes[0].text(rt, max(signal)*0.8, name + " (putative)", rotation=90, va="top")
    axes[1].axis("off")
    axes[1].text(0, 1, filename + "\nAutomatic integration candidates: review required. Not TFV/FTC.", fontsize=8, va="top")
    axes[0].set_xlabel("Time (min)")
    fig.tight_layout()
    fig.savefig(out / "qc_real_chromatogram.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/external/chromatogram_downloads.json"))
    p.add_argument("--summary", type=Path, default=Path("data/external/uliege_csm5gsmr5t/Summary.xlsx"))
    p.add_argument("--out", type=Path, default=Path("data/processed/public_retention/chromatograms"))
    main(p.parse_args())
