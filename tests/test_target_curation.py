import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_open_methods import extract_evidence
from curate_target_evidence import retention_conflict, doe_candidates, identity_status


class TargetCurationTests(unittest.TestCase):
    def test_connectivity_match_does_not_approve_different_stereochemistry(self):
        self.assertEqual(identity_status("XQSPYNMVSIKCOC-NTSWFWBYSA-N", "emtricitabine"), "exact_full_inchikey")
        self.assertNotEqual(identity_status("XQSPYNMVSIKCOC-PRJDIBJQSA-N", "emtricitabine"), "exact_full_inchikey")
        self.assertNotEqual(identity_status("SGOIRFVFHAKUTI-UHFFFAOYSA-N", "tenofovir"), "exact_full_inchikey")

    def test_doe_unicode_minus_and_explicit_center_ph_mapping(self):
        rows = ''.join(f'<tr><td>{i}</td><td>{i}</td><td>−1.00</td><td>0.00</td>'
                       '<td>4.76</td><td>7.64</td><td>1.1</td></tr>' for i in range(1, 14))
        root = ET.fromstring('<article><floats-group><table-wrap id="tab2"><table>' + rows +
                             '</table></table-wrap></floats-group></article>')
        records = doe_candidates(root, {})
        self.assertEqual(len(records), 13)
        self.assertEqual(records[0]["methanol_percent"], 10)
        self.assertEqual(records[0]["reported_mobile_phase_pH"], 3.5)
        self.assertFalse(records[0]["training_ready"])
        self.assertFalse(records[0]["target_pair"])

    def test_floating_tables_and_nested_captions_are_not_lost(self):
        root = ET.fromstring('''<article><body><p>Methods.</p></body><floats-group>
        <table-wrap id="t1"><label>Table 1</label><caption><p>Retention <italic>times</italic></p></caption>
        <table><tr><th>Emtricitabine</th><td>4.2</td></tr></table></table-wrap>
        <table-wrap id="t2"><table><tr><td>Other</td></tr></table></table-wrap>
        </floats-group></article>''')
        evidence, tables = extract_evidence(root, "PMC1", {}, "https://example.org")
        self.assertEqual(len(tables), 2)  # Keep even tables not matched by topic regex.
        self.assertEqual(tables[0]["caption"], "Table 1 Retention times")
        self.assertEqual(tables[0]["rows"][0][1]["text"], "4.2")
        self.assertFalse(tables[0]["training_ready"])
        self.assertEqual(evidence[0]["node_id"], "t1")

    def test_conflicting_author_times_stay_separate_and_blocked(self):
        root = ET.fromstring('''<article><body><p>The retention times for lamivudine,
        telbivudine, emtricitabine, entecavir, tenofovir, nedaplatin, oxaliplatin,
        adefovir dipivoxil and IS were found to be 5.60, 5.97, 6.21, 5.84, 5.08,
        7.91, 8.74, 7.19, and 7.70 min.</p></body><floats-group><table-wrap id="x-t005">
        <table><tr><th>Compound</th><th>Retention Time (min)</th></tr>
        <tr><td>Emtricitabine</td><td>0.78</td></tr>
        <tr><td>Tenofovir</td><td>0.80</td></tr></table></table-wrap></floats-group></article>''')
        result = retention_conflict(root)
        self.assertEqual(result["narrative_tr_min"]["tenofovir"], 5.08)
        self.assertEqual(result["table_tr_min"]["tenofovir"], 0.80)
        self.assertEqual(len(result["conflicting_analytes"]), 2)
        self.assertFalse(result["training_ready"])


if __name__ == "__main__":
    unittest.main()
