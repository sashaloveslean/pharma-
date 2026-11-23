import math
from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import subprocess
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_chromatogram_peaks import half_height_width, rs_half_height, filename_ph, measure_peak, read_dad
from collect_open_methods import license_allowed
from collect_public_data import fetch
from import_reported_suitability import parse_suitability


class SignalExtractionTests(unittest.TestCase):
    def test_reported_missing_resolution_is_not_replaced_with_zero(self):
        rows = [["Parameters", "Emtricitabine (FTC)", "Daclatasvir (DAC)", "Ledipasvir (LDV)"],
                ["Retention time (min)", "2.60", "4.80", "9.89"],
                ["Tailing factor (T)", "1.33", "1.36", "1.15"],
                ["Number of theoretical plates (N)", "2578", "3366", "3326"],
                ["Capacity factor (k’)", "1.17", "3", "7.25"],
                ["Resolution (R s )", "−", "4.73", "7.32"]]
        table = {"rows": [[{"text": str(v)} for v in row] for row in rows]}
        parsed = parse_suitability(table)
        self.assertIsNone(parsed[0]["reported_resolution_for_peak"])
        self.assertEqual(parsed[1]["reported_resolution_for_peak"], 4.73)
        self.assertIsNone(parsed[0]["measured_t0_min"])
        table["rows"][0][0]["text"] = "Parameters RSD (%)"
        with self.assertRaises(ValueError):
            parse_suitability(table)

    def test_interrupted_download_resumes_and_verifies_before_publish(self):
        content = b"original-public-signal"
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/"signal.arw"
            def fake_run(command):
                partial = Path(command[-1])
                calls.append(command)
                if len(calls) == 1:
                    partial.write_bytes(content[:7])
                    self.assertFalse(destination.exists())
                    return subprocess.CompletedProcess(command, 28)
                self.assertEqual(partial.read_bytes(), content[:7])
                partial.write_bytes(content)
                return subprocess.CompletedProcess(command, 0)
            with patch("collect_public_data.subprocess.run", side_effect=fake_run):
                record = fetch("https://example.org/signal", destination, hashlib.sha256(content).hexdigest())
            self.assertEqual(destination.read_bytes(), content)
            self.assertEqual(record["bytes"], len(content))
            self.assertTrue(all("-C" in command for command in calls))

    def test_quoted_and_unquoted_numeric_exports_have_identical_values(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = ["Wavelength,220,260", "Time"] + [f"{i/10},{i},{2*i}" for i in range(20)]
            left, right = Path(directory)/"plain.arw", Path(directory)/"quoted.arw"
            left.write_text("\n".join(plain))
            right.write_text("\n".join('"'+line+'"' if i!=1 else line for i,line in enumerate(plain)))
            for a,b in zip(read_dad(left), read_dad(right)):
                np.testing.assert_array_equal(a,b)

    def test_half_height_uses_actual_baseline_corrected_height(self):
        x = np.linspace(0, 10, 10001)
        y = np.exp(-0.5*((x-5)/0.2)**2)
        width, left, right = half_height_width(x, y, np.argmax(y))
        self.assertAlmostEqual(width, 2*math.sqrt(2*math.log(2))*0.2, places=5)
        self.assertAlmostEqual((left+right)/2, 5)

    def test_local_baseline_removes_constant_offset_and_linear_drift(self):
        x = np.linspace(0, 10, 10001)
        y = 3 + 0.2*x + np.exp(-0.5*((x-5)/0.05)**2)
        result = measure_peak(x, y, [5])
        self.assertEqual(result["status"], "derived_needs_review")
        self.assertAlmostEqual(result["width_half_height_min"], 2.35482*0.05, places=3)
        self.assertFalse(result["training_ready"])

    def test_resolution_is_half_height_formula_not_baseline_formula(self):
        a = {"retention_time_min": 2, "width_half_height_min": 0.2}
        b = {"retention_time_min": 3, "width_half_height_min": 0.3}
        self.assertAlmostEqual(rs_half_height(a,b), 2.36)

    def test_filename_decimal_codes_are_mapped_explicitly(self):
        self.assertEqual(filename_ph("35"), 3.5)
        self.assertEqual(filename_ph("65"), 6.5)
        self.assertEqual(filename_ph("8"), 8)

    def test_publicly_accessible_does_not_mean_permissive_license(self):
        for path, expected in [("by/4.0", True), ("by-nc/4.0", False), ("by-nd/4.0", False)]:
            root = ET.fromstring(f'<article><permissions><license href="https://creativecommons.org/licenses/{path}/"/></permissions></article>')
            self.assertEqual(license_allowed(root)[0], expected)
        self.assertFalse(license_allowed(ET.fromstring('<article/>'))[0])


if __name__ == "__main__":
    unittest.main()
