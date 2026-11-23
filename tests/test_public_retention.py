"""Regression checks for scientific units and normalized method overrides."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_public_retention import smrt
from open_retention import load_observations


class PublicRetentionTests(unittest.TestCase):
    def test_smrt_seconds_are_converted_and_invalid_labels_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "metlin_smrt"
            source.mkdir()
            (source / "source_metadata.json").write_text(json.dumps({
                "url": "https://example.org", "license": {"name": "Apache 2.0"}}))
            (source / "SMRT_dataset.csv").write_text(
                'pubchem;rt;inchi\n962;90;InChI=1S/H2O/h1H2\n'
                '962;0;InChI=1S/H2O/h1H2\n0;120;invalid\n')
            smrt(root, root / "out")
            rows = load_observations(root / "out/metlin_smrt/observations.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["retention_time_min"], 1.5)
            self.assertEqual(rows[0]["original_rt_unit"], "s")
            self.assertIsNone(rows[0]["run_id"])
            self.assertIsNone(rows[0]["Rs"])
            report = json.loads((root / "out/metlin_smrt/quality_report.json").read_text())
            self.assertEqual(report["issues"]["invalid_inchi_or_rt"], 2)

    def test_per_observation_ph_overrides_shared_system_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "systems.json").write_text(json.dumps({"a": {
                "metadata": {"pH": 3}, "gradient_program": [{"time": 0}]}}))
            path = root / "observations.jsonl"
            path.write_text(json.dumps({"system_id": "a", "metadata": {"pH": 7}}) + "\n")
            row = load_observations(path)[0]
            self.assertEqual(row["metadata"]["pH"], 7)
            self.assertEqual(row["gradient_program"], [{"time": 0}])


if __name__ == "__main__":
    unittest.main()
