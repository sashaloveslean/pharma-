#!/usr/bin/env python3
"""Import explicitly reported suitability values, separately from signal-derived Rs."""
import hashlib
import json
import math
from pathlib import Path


def parse_suitability(table):
    grid = [[cell["text"] for cell in row] for row in table["rows"]]
    if not grid or any("RSD" in cell.upper() for cell in grid[0]):
        raise ValueError("RSD robustness percentages are not absolute resolution")
    expected = ("Emtricitabine (FTC)", "Daclatasvir (DAC)", "Ledipasvir (LDV)")
    if tuple(grid[0][1:4]) != expected:
        raise ValueError("Unexpected analyte headers")
    values = {row[0].strip(): row[1:4] for row in grid[1:]}
    rows = []
    for index, name in enumerate(expected):
        k = float(values["Capacity factor (k’)" ][index])
        rs = values["Resolution (R s )"][index]
        rows.append({"compound": name, "reported_tr_min": float(values["Retention time (min)"][index]),
                     "reported_tailing_factor": float(values["Tailing factor (T)"][index]),
                     "reported_theoretical_plates": float(values["Number of theoretical plates (N)"][index]),
                     "reported_retention_factor": k, "log10_k_from_reported_k": math.log10(k),
                     "reported_resolution_for_peak": None if rs in {"−", "-"} else float(rs),
                     "measured_t0_min": None, "reported_peak_width_min": None})
    return rows


if __name__ == "__main__":
    root = Path("data/external/open_methods")
    out = Path("data/processed/public_retention/reported_suitability")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    entry = next(r for r in manifest["articles"] if r["pmcid"] == "PMC12216858")
    if entry["status"] != "downloaded":
        raise SystemExit("Source license is not confirmed")
    tables = json.loads((root / "PMC12216858_tables.json").read_text())
    table = next(t for t in tables if t["node_id"] == "Tab8")
    records = parse_suitability(table)
    method = {"mode": "RP-isocratic", "column": "Exsil 100 ODS C18", "column_length_mm": 250,
              "column_id_mm": 4.6, "particle_um": 5, "flow_ml_min": 1,
              "temperature_c": 25, "temperature_basis": "reported room temperature",
              "mobile_phase": {"acetonitrile_percent": 20, "methanol_percent": 60,
                               "ammonium_acetate_buffer_percent": 20},
              "aqueous_buffer_concentration_mM": 10, "aqueous_buffer_pH": 4.5,
              "mixed_mobile_phase_pH": None, "injection_ul": 20, "detector": "programmed fluorescence",
              "fluorescence_program_source_table": "Tab1", "run_time_min": 15,
              "method_record_id": "PMC12216858:method1", "run_id": None,
              "source_url": entry["url"], "doi": entry["doi"], "license_evidence": entry["license_evidence"],
              "source_sha256": hashlib.sha256((root / "PMC12216858.xml").read_bytes()).hexdigest()}
    (out / "method.json").write_text(json.dumps(method, indent=2))
    with (out / "reported_peak_parameters.jsonl").open("w") as file:
        for record in records:
            file.write(json.dumps({**record, "method_record_id": method["method_record_id"],
                                   "source_table": "Tab8", "source_url": entry["url"],
                                   "origin": "author_reported_method_summary", "training_ready": False}) + "\n")
    pairs = []
    for a,b in zip(records, records[1:]):
        pairs.append({"compound_1": a["compound"], "compound_2": b["compound"],
                      "reported_rs": b["reported_resolution_for_peak"], "method_record_id": method["method_record_id"],
                      "source_table": "Tab8", "source_url": entry["url"], "run_id": None,
                      "origin": "author_reported_resolution_method_summary",
                      "pair_assignment": "adjacent peaks in table/elution order; confirm with chromatogram",
                      "reviewed": False, "training_ready": False})
    (out / "reported_resolution.json").write_text(json.dumps(pairs, indent=2))
    print(json.dumps({"reported_peak_records": len(records), "reported_resolution_values": len(pairs),
                      "method": method["method_record_id"], "note": "Not a TFV/FTC separation"}, indent=2))
