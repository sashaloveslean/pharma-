"""Condition/molecule interactions; geometric volume/F is explicitly NOT measured t0."""
from functools import lru_cache
import math
from rdkit import Chem
from rdkit.Chem import Descriptors
from method_physics import bed_volume_ml, linear_velocity_mm_s


@lru_cache(maxsize=50000)
def polarity(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid molecular structure')
    mw=Descriptors.MolWt(mol);heavy=mol.GetNumHeavyAtoms()
    return {'tpsa_per_mw':Descriptors.TPSA(mol)/mw,
            'formal_charge':Chem.GetFormalCharge(mol),
            'hetero_fraction':sum(a.GetAtomicNum() not in (1,6) for a in mol.GetAtoms())/max(heavy,1),
            'hbond_per_heavy_atom':(Descriptors.NumHDonors(mol)+Descriptors.NumHAcceptors(mol))/max(heavy,1),
            'fraction_sp3':Descriptors.FractionCSP3(mol)}


def physical_features(row):
    m=row['metadata'];p=row['gradient_program']
    volume=bed_volume_ml(m['column.length'],m['column.id']);flow=p[0]['flow rate [ml/min]']
    if not math.isfinite(flow) or flow<=0:raise ValueError('Positive finite flow required')
    organic=[sum(point[f'{ch} [%]']*(m[f'eluent.{ch}.meoh']+m[f'eluent.{ch}.acn'])/100 for ch in 'AB') for point in p]
    stop=next(i for i,value in enumerate(organic) if value==max(organic))
    end=p[stop]['t [min]']
    if end<=0 or organic[stop]<=organic[0]:raise ValueError('Increasing gradient required')
    ramp_volume=sum((b['t [min]']-a['t [min]'])*(a['flow rate [ml/min]']+b['flow rate [ml/min]'])/2
                    for a,b in zip(p[:stop],p[1:stop+1]))
    departure=next(i for i,value in enumerate(organic) if value!=organic[0])
    initial_hold=p[departure-1]['t [min]']
    polar=polarity(row['smiles'])
    initial_meoh=sum(p[0][f'{ch} [%]']*m[f'eluent.{ch}.meoh']/100 for ch in 'AB')/100
    initial_acn=sum(p[0][f'{ch} [%]']*m[f'eluent.{ch}.acn']/100 for ch in 'AB')/100
    result={'geometric_bed_volume_ml':volume,'geometric_volume_over_flow_min':volume/flow,
            'superficial_velocity_mm_s':linear_velocity_mm_s(flow,m['column.id']),
            'geometric_time_over_analytical_time':volume/flow/end,
            'analytical_volume_over_bed_volume':ramp_volume/volume,
            'initial_hold_over_analytical_time':initial_hold/end,
            'mean_organic_slope_fraction_per_min':(organic[stop]-organic[0])/100/end,
            'initial_meoh_fraction':initial_meoh,'initial_acn_fraction':initial_acn,**polar,
            'polarity_x_initial_meoh':polar['tpsa_per_mw']*initial_meoh,
            'polarity_x_initial_acn':polar['tpsa_per_mw']*initial_acn,
            'polarity_x_geometric_time':polar['tpsa_per_mw']*volume/flow}
    for additive in ('formic','nh4form','nh4ac'):
        result['initial_'+additive]=sum(p[0][f'{ch} [%]']*m[f'eluent.{ch}.{additive}']/100 for ch in 'AB')
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Nonfinite physical features')
    return result
