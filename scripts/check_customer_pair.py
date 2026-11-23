"""Run frozen research retention model on the unchanged customer candidate."""
import hashlib
import json
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from rdkit import Chem
from evaluate_external_transfer import adapt_system
from evaluate_mcmrt_transfer import feature, analytical_scale


def run():
    model_path = Path('reports/external_transfer/research_scaled.cbm')
    systems = json.loads(Path('data/processed/public_retention/report/systems.json').read_text())
    recommendation = json.loads(Path('reports/customer_multicolumn/lab_recommendation.json').read_text())
    candidate = recommendation['primary']['candidate']
    if candidate['candidate_id'] != 'report:0386:time_x1.0':
        raise ValueError('Selected customer candidate changed; review the run configuration')
    system = adapt_system(systems['report:0386'])
    if system['gradient_program'] != [{k: float(p[k]) for k in system['gradient_program'][0]}
                                     for p in candidate['gradient_program']]:
        raise ValueError('Selected gradient and source disagree')
    model = CatBoostRegressor(); model.load_model(str(model_path))
    expected = {'tenofovir': 'SGOIRFVFHAKUTI-ZCFIWIBFSA-N',
                'emtricitabine': 'XQSPYNMVSIKCOC-NTSWFWBYSA-N'}
    predictions = []
    for name, key in expected.items():
        cached = json.loads(Path(f'reports/tenofovir_demo/pubchem_{name}.json').read_text())
        compound = cached['payload']['PropertyTable']['Properties'][0]
        if compound['InChIKey'] != key or Chem.MolToInchiKey(Chem.MolFromSmiles(compound['SMILES'])) != key:
            raise ValueError('Target identity disagreement')
        row = {**system, 'smiles': compound['SMILES']}
        predicted = max(0., float(np.expm1(model.predict([feature(row)])[0]))) * analytical_scale(row)
        predictions.append({'name': name, 'inchikey': key, 'smiles': compound['SMILES'],
                            'predicted_tr_min': predicted, 'identity_source': cached['source_url']})
    calibration = json.loads(Path('reports/local_calibration/evaluation.json').read_text())
    result = {'candidate_id': candidate['candidate_id'], 'system_id': 'report:0386',
              'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'target': 'log1p(tR / time_to_first_maximum_organic)',
              'metadata': system['metadata'], 'gradient_program': system['gradient_program'],
              'predictions': predictions, 'local_calibration_applied': False,
              'calibration_reason': 'No independent calibration measurements for the selected complete method; offsets from other methods cannot be transferred.',
              'predicted_delta_tr_min': abs(predictions[0]['predicted_tr_min']-predictions[1]['predicted_tr_min']),
              'predicted_Rs': None, 'separation_confirmed': False, 'validated_for_customer_method': False,
              'status': 'research_prediction_requires_experiment',
              'external_target_diagnostics_other_methods': calibration['target_diagnostics'],
              'limitations': ['Frozen research model trained on MCMRT; not a replacement of the production model.',
                              'No measured TFV/FTC validation for report:0386 in this run.',
                              'No calibrated uncertainty, measured t0, peak widths or Rs.',
                              'Predicted time gap does not establish separation or elution order.',
                              'External FTC diagnostic is a different column/program; no transfer of its offset.']}
    out = Path('reports/customer_pair_check'); out.mkdir(parents=True, exist_ok=True)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Проверка TFV + FTC', '', 'Исследовательский прогноз на неизменённом методе report:0386.',
             f"Модель: {model_path}", '', '| Вещество | Прогноз tR, мин |', '|---|---:|']
    for r in predictions: lines.append(f"| {r['name']} | {r['predicted_tr_min']:.3f} |")
    lines += ['', 'Локальная калибровка не применена: нет измерений для этого полного метода.',
              'Разделение не подтверждено. Разность времён не является Rs.', *result['limitations']]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': run()
