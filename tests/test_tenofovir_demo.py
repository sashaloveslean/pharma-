import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from demo_tenofovir import EXCLUDED, eligible


class TargetHoldoutTests(unittest.TestCase):
    def test_targets_and_prodrugs_never_enter_training_even_under_synonyms(self):
        for prefix in EXCLUDED:
            self.assertFalse(eligible({"compound_id": prefix + "-STEREO-N", "name": "synonym"}))

    def test_named_salt_is_excluded_even_with_different_connectivity_key(self):
        self.assertFalse(eligible({"compound_id": "OTHER-STEREO-N", "name": "Tenofovir disoproxil fumarate"}))
        self.assertTrue(eligible({"compound_id": "OTHER-STEREO-N", "name": "ibuprofen"}))


if __name__ == "__main__":
    unittest.main()
