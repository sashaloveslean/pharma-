"""Proposed TFV/FTC standard/DAD experiment plan; no unvalidated ML phase ranking."""
import json
from pathlib import Path
from copy import deepcopy

OUT=Path('reports/lab_plan_revision_2026_11_09')

def build():
    rnd=json.loads(Path('reports/rnd_gradient_design/result.json').read_text())
    phases=[
        dict(id='P0',kind='source_control',salt=None,salt_added_mM=0,aqueous_target_pH=None,
             A='Water + 0.1% v/v formic acid',B='Methanol + 0.1% v/v formic acid',source_pH=2.6),
        dict(id='P1',kind='proposed_buffer',salt='ammonium formate',salt_added_mM=10,aqueous_target_pH=3.0,
             A='10 mM added ammonium formate in water, adjust with formic acid to pH 3.0',B='Methanol without added salt or acid'),
        dict(id='P2',kind='proposed_buffer',salt='ammonium formate',salt_added_mM=10,aqueous_target_pH=4.0,
             A='10 mM added ammonium formate in water, adjust with formic acid to pH 4.0',B='Methanol without added salt or acid'),
        dict(id='P3',kind='proposed_buffer',salt='ammonium acetate',salt_added_mM=10,aqueous_target_pH=5.0,
             A='10 mM added ammonium acetate in water, adjust with acetic acid to pH 5.0',B='Methanol without added salt or acid')]
    baseline=deepcopy(rnd['baseline']['gradient_program'])
    candidates=[]
    for p in phases:
        candidates.append(dict(id=p['id']+'_G0',phase_id=p['id'],gradient_program=baseline,
            origin=p['kind'],predicted_tr_min=None,predicted_Rs=None,model_rank=None,
            pH_effect_model_validated=False,training_approved=False))
    criteria={'status':'proposed_RnD_targets_not_compendial_validation',
        'resolution_half_height_min':1.5,'preferred_resolution_min':2.0,
        'tailing_at_5pct_range':[0.8,2.0],'retention_time_rsd_pct_max':1.0,
        'peak_area_rsd_pct_max':2.0,'confirmation_injections':6,
        'signal_to_noise_min':10,'carryover_pct_of_10ug_ml_peak_area_max':0.1,
        'blank_interference_pct_of_10ug_ml_peak_area_max':0.1,
        'require_unsaturated_signal':True,'require_identified_peaks':True,
        'no_missing_measurement_pass':True}
    plan={'version':'customer_feedback_v1','date':'2026-11-09',
        'assumptions':{'sample':'pure standards TFV and FTC, no matrix','detector':'UV/DAD, assumed pending customer confirmation'},
        'targets':['TFV, not TDF or TAF','FTC'],
        'fixed':{'column':'Acclaim RSLC 120 C18','length_mm':100,'id_mm':2.1,'particle_um':2.2,
                 'temperature_C':30,'flow_ml_min':0.2,'injection_uL':1,
                 'DAD_range_nm':[200,320],'common_channel_nm':260,'additional_FTC_channel_nm':280},
        'phases':phases,'stage1_candidates':candidates,
        'stage2':{'phase':'selected from measured stage1 results','gradients':[r['gradient_program'] for r in rnd['selected'][:2]],
                  'selection_rule':'Pass measured criteria, then minimize cycle time; do not select by predicted gap alone'},
        'concentration_followup':{'same_salt_same_pH':True,'salt_added_mM':[5,10,20],
                                'only_if_needed':True,'do_not_change_gradient_simultaneously':True},
        'criteria':criteria,
        'prep':{'diluent':'For each phase, pre-mix its A:B 90:10 v/v; verify clarity',
                'stocks_ug_ml':100,'stock_mass_mg_at_100pct_purity':10,'stock_final_volume_ml':100,
                'purity_correction':'mass = 10 / certified_analyte_mass_fraction; account for salt/hydrate basis',
                'single_standard_ug_ml':10,'mixture_levels_ug_ml_each':[5,10,20],
                'mixture_stock_aliquots_ml_each':[0.5,1,2],'mixture_final_volume_ml':10,
                'stock_release':'Fully dissolved and stable over working period; do not silently change diluent or filter away undissolved analyte',
                'stability':'Fresh on day of experiment; storage and matrix extraction not validated'},
        'sequence':{'per_phase_stage1':['diluent blank','TFV 10 ug/mL','FTC 10 ug/mL','mixture 10 each, repeat 3 times','post-mixture blank'],
                    'stage1_injections':28,'stage1_cycle_time_min_excluding_equilibration':840,
                    'confirmation':'Best phase/gradient: 6 consecutive mixture injections at 10 ug/mL each; then 5 and 20 levels and a carryover blank after 20',
                    'equilibration':'Do not assume fixed return time sufficient; require stable pressure/baseline and reproducible standard tR'},
        'buffer_notes':['10 mM denotes added salt in reservoir A, not total conjugate acid+base after titration.',
                        'Salt varies with A fraction: 9 mM at 10%B to 0.5 mM at 95%B; aqueous pH is not constant mixed-solvent pH.',
                        'Check clarity at A:B 90:10, 50:50 and 5:95, including transitions, before connecting column.',
                        'Remove salts with compatible aqueous-organic flush before strong-organic cleaning; do not introduce pure water into column beyond its aqueous limit.',
                        'Phase P0 has acid in both reservoirs; buffer candidates differ in additives as well as pH. This is screening, not isolation of pH effect.'],
        'instrument_execution_authorized':False,
        'ML_limit':'Current model has no validated pH response or peak shape predictor. New phase candidates are rules-based experiments, not a trained optimum.'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'experiment_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    template={'condition_id':None,'phase_id':None,'actual_aqueous_pH':None,'salt_added_mM':None,
              'actual_gradient_program':None,'sample_id':None,'injection_number':None,'detector_channel_nm':260,
              'concentration_ug_ml_each':None,'injection_uL':None,'pressure_max_bar':None,
              'TFV':{'identity_confirmed':None,'tr_min':None,'width_half_height_min':None,'tailing_5pct':None,'area':None,'snr':None},
              'FTC':{'identity_confirmed':None,'tr_min':None,'width_half_height_min':None,'tailing_5pct':None,'area':None,'snr':None},
              'Rs_measured':None,'raw_chromatogram_path':None,'outcome':None,'label_origin':'experiment','reviewed':False,'training_approved':False}
    (OUT/'measurement_template.json').write_text(json.dumps(template,indent=2)+'\n')
    return plan

if __name__=='__main__':
    p=build();print(json.dumps({'phases':len(p['phases']),'stage1_injections':28,'output':str(OUT)},indent=2))
