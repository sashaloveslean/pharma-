import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_lss_identities import decide
from train_lss_parameters import molecule_folds


class IdentityTests(unittest.TestCase):
    def response(self):
        return {"PropertyTable":{"Properties":[{"SMILES":"CCO","InChIKey":"LFQSCWFLJHTTHZ-UHFFFAOYSA-N","CID":702}]}}

    def test_agreement_is_not_source_vial_verification(self):
        r=decide("ethanol",self.response(),{"full_inchikey":"LFQSCWFLJHTTHZ-UHFFFAOYSA-N"})
        self.assertEqual(r["smiles"],"CCO")
        self.assertFalse(r["source_identity_reviewed"])

    def test_conflicting_identity_excluded(self):
        r=decide("ethanol",self.response(),{"full_inchikey":"other-key-N"})
        self.assertIsNone(r["smiles"])
        self.assertEqual(r["status"],"local_pubchem_identity_conflict")

    def test_bad_structure_and_multiple_hits_excluded(self):
        r=self.response();r["PropertyTable"]["Properties"][0]["SMILES"]="CC"
        self.assertIsNone(decide("ethanol",r,{})["smiles"])
        r=self.response();r["PropertyTable"]["Properties"]*=2
        self.assertIsNone(decide("ethanol",r,{})["smiles"])
        self.assertIsNone(decide("typo",{}, {})["smiles"])

    def test_old_fold_assignments_preserved_after_expansion(self):
        old=[str(i) for i in range(10) for _ in range(3)]
        original={old[i]:fold for fold,_,test in molecule_folds(old) for i in test}
        expanded=old+["new"]*5
        seen=[]
        for fold,train,test in molecule_folds(expanded,old):
            self.assertFalse({expanded[i] for i in train}&{expanded[i] for i in test})
            seen.extend(test)
            for i in test:
                if expanded[i] in original:self.assertEqual(fold,original[expanded[i]])
        self.assertEqual(sorted(seen),list(range(len(expanded))))
