#!/usr/bin/env python3
"""Predict one molecule or a pair on an imported public chromatographic system."""
import argparse
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from train_open_retention import vector
from open_retention import load_observations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", required=True)
    p.add_argument("--second-smiles")
    p.add_argument("--system", required=True, help="For example report:0001")
    p.add_argument("--data", type=Path, default=Path("data/processed/open_retention/observations.jsonl"))
    p.add_argument("--model-dir", type=Path, default=Path("models/open_retention"))
    args = p.parse_args()
    evaluation = json.loads((args.model_dir / "evaluation.json").read_text())
    rows = load_observations(args.data)
    rows = [r for r in rows if r["source_info"][0].get("method.type") == evaluation["mode"]]
    if sorted({r["source_revision"] for r in rows}) != evaluation["source_revisions"]:
        p.error("Dataset version differs from model; rebuild the model")
    reference = next((r for r in rows if r["system_id"] == args.system), None)
    if reference is None:
        p.error("Unknown system: no validated transfer; import its measurements first")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in rows]
    model = CatBoostRegressor()
    model.load_model(str(args.model_dir / "catboost.cbm"))
    predictions = []
    for smiles in filter(None, [args.smiles, args.second_smiles]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            p.error("Invalid SMILES")
        similarities = DataStructs.BulkTanimotoSimilarity(generator.GetFingerprint(mol), fingerprints)
        local = max(s for s, r in zip(similarities, rows) if r["system_id"] == args.system)
        row = {**reference, "smiles": smiles}
        predicted = max(0, float(np.expm1(model.predict([vector(row, evaluation["numeric_metadata_features"])])[0])))
        predictions.append({"smiles": smiles, "predicted_tr_min": round(predicted, 3),
                            "max_local_tanimoto_128bit": round(local, 3),
                            "needs_experiment": True,
                            "validation_mae_min": evaluation["validation"]["new_molecule"]["per_system"][args.system]["mae_min"]})
    result = {"system_id": args.system, "predictions": predictions,
              "system_conditions": reference["metadata"], "gradient_program": reference["gradient_program"],
              "source_url": reference["source_url"], "predicted_Rs": None,
              "status": "screening_estimate_requires_experimental_validation"}
    if len(predictions) == 2:
        result["predicted_delta_tr_min"] = round(abs(predictions[1]["predicted_tr_min"] - predictions[0]["predicted_tr_min"]), 3)
        result["separation_confirmed"] = False
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
