"""Import only experimental sheets 1–3 of licensed MCMRT supplementary data.

Sheets containing projections, calibration outputs and classifications are never
used as prediction labels/features. Method conflicts are quarantined explicitly.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from openpyxl import load_workbook
from rdkit import Chem, rdBase


def clean(value):
    return ' '.join(str(value or '').split())


def dimensions(text):
    values=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',text)]
    if len(values)!=3:raise ValueError('Unknown column dimensions')
    a,b,particle=values
    # Source uses both L x ID and ID x L. Preserve original text alongside mapping.
    length,diameter=max(a,b),min(a,b)
    if not 20<=length<=300 or not .1<=diameter<=10 or not .5<=particle<=20:
        raise ValueError('Unsupported column geometry')
    return length,diameter,particle


def phase(text):
    t=clean(text).lower()
    result={'h2o':0.,'meoh':0.,'acn':0.,'formic':0.,'nh4form':0.,'nh4ac':0.}
    if 'water/methanol=90:10' in t:result.update(h2o=90.,meoh=10.)
    elif t.startswith('water'):result['h2o']=100.
    elif t.startswith('methanol'):result['meoh']=100.
    elif t.startswith('acetonitrile'):result['acn']=100.
    else:raise ValueError('Unknown solvent recipe')
    if 'formic acid' in t:
        m=re.search(r'([\d.]+)%\s*formic acid',t)
        if not m:raise ValueError('Unknown acid concentration')
        result['formic']=float(m[1])
    for word,key in [('ammonium formate','nh4form'),('ammonium acetate','nh4ac')]:
        if word in t:
            m=re.search(r'([\d.]+)\s*mm\s*'+word,t)
            if not m:raise ValueError('Unknown buffer concentration')
            result[key]=float(m[1])
    return result


def build(root,out):
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/'Supplementary Data 1-14.xlsx'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:raise ValueError('Source checksum mismatch')
    if manifest['license']!='CC BY 4.0':raise ValueError('Unexpected supplement license')
    book=load_workbook(path,read_only=True,data_only=True)
    formulas=load_workbook(path,read_only=True,data_only=False)
    issues=[];molecules={};rows=[];systems={};missing_outcomes=[]
    with rdBase.BlockLogs():
        for index,row in enumerate(book['1'].iter_rows(min_row=2,values_only=True),2):
            cid,name,smiles,inchi=row[0],clean(row[1]),clean(row[6]),clean(row[7])
            if not isinstance(cid,int):continue
            mol=Chem.MolFromSmiles(smiles)
            if mol is None or Chem.MolToInchiKey(mol)!=Chem.InchiToInchiKey(inchi):
                issues.append({'kind':'structure_inchi_disagreement','compound_number':cid,'name':name});continue
            molecules[cid]={'compound_id':Chem.MolToInchiKey(mol),'smiles':Chem.MolToSmiles(mol,isomericSmiles=True),
                            'name':name,'source_compound_number':cid,'identity_evidence_cell':f"1!G{index}:H{index}"}
    matrix=list(book['2'].values)
    recipes={}
    for row in matrix[20:]:
        if row[0] and re.match(r'^[AB]\d:',row[0]):
            code,text=row[0].split(':',1);recipes[code]=phase(text)
    for j in range(30):
        c=1+3*j;sid=f'mcmrt:{j+1:02d}'
        if clean(matrix[0][c])!=f'CM {j+1:02d}':raise ValueError('Unexpected method header')
        original={clean(row[0]):row[c] for row in matrix[:12]}
        name=clean(matrix[6][c]);geometry=clean(matrix[7][c]);L,D,P=dimensions(geometry)
        meta={'column.name':name,'column.usp.code':'L1','column.length':L,'column.id':D,
              'column.particle.size':P,'column.temperature':float(matrix[9][c]),'column.t0':None}
        conflicts=[]
        recipe_code=re.search(r'(A\d+)(B\d+)',str(matrix[1][c]))
        for channel,offset in [('A',10),('B',11)]:
            parsed=phase(matrix[offset][c])
            if recipe_code and parsed!=recipes.get(recipe_code[1 if channel=='A' else 2]):
                conflicts.append('recipe_code_disagrees_with_explicit_phase_'+channel)
            for key,value in parsed.items():meta[f'eluent.{channel}.{key}']=value
            meta[f'eluent.{channel}.pH']=None
        program=[]
        for raw in matrix[13:20]:
            time,flow,b=raw[c:c+3]
            if time is None and flow is None and b is None:continue
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (time,flow,b)):
                raise ValueError('Incomplete gradient triple')
            program.append({'t [min]':float(time),'flow rate [ml/min]':float(flow),
                            'A [%]':100-float(b),'B [%]':float(b),'C [%]':0.,'D [%]':0.})
        if not program or program[0]['t [min]']!=0:raise ValueError('Incomplete gradient')
        if any(b['t [min]']<=a['t [min]'] for a,b in zip(program,program[1:])):raise ValueError('Nonincreasing gradient')
        if any(p['flow rate [ml/min]']<=0 or not 0<=p['B [%]']<=100 for p in program):raise ValueError('Invalid flow/composition')
        meta['column.flowrate']=program[0]['flow rate [ml/min]']
        systems[sid]={'metadata':meta,'gradient_program':program,'source_method':original,
            'source_geometry_text':geometry,'source_info':[{'method.type':'RP','url':manifest['article'],'name':f'CM {j+1:02d}'}],
            'issues':conflicts,'eligible_for_rt_research':not conflicts,'t0_provenance':'absent_in_this_supplement',
            'instrument':clean(matrix[5][c]),'source_sheet':'2','source_column':c+1}
        issues.extend({'kind':issue,'system_id':sid} for issue in conflicts)
    formula_rows=list(formulas['3'].values)
    for i,row in enumerate(book['3'].iter_rows(min_row=2,values_only=True),2):
        cid=row[0]
        if not isinstance(cid,int):continue  # Trailing annotation/stray number is not a compound.
        if cid not in molecules:continue
        if clean(row[1]).casefold()!=molecules[cid]['name'].casefold():raise ValueError('Name/compound number mismatch')
        for j in range(30):
            value=row[8+j];sid=f'mcmrt:{j+1:02d}';formula=formula_rows[i-1][8+j]
            if isinstance(formula,str) and formula.startswith('='):
                issues.append({'kind':'formula_label_excluded','system_id':sid,'compound_number':cid});continue
            if isinstance(value,str) and value.strip().lower()=='n.d':
                missing_outcomes.append({'system_id':sid,'compound_id':molecules[cid]['compound_id'],
                                         'compound_number':cid,'status':'not_detected',
                                         'retention_time_min':None,'source_cell':f"3!{book['3'].cell(1,9+j).column_letter}{i}"})
                continue
            if value in (None,'','-'):continue
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
                issues.append({'kind':'invalid_rt','system_id':sid,'compound_number':cid});continue
            if value>systems[sid]['gradient_program'][-1]['t [min]']:
                issues.append({'kind':'rt_after_program','system_id':sid,'compound_number':cid});continue
            rows.append({**molecules[cid],'observation_id':f'{sid}:{cid}','system_id':sid,'retention_time_min':float(value),
                         'rt_unit':'min','label_origin':'experiment','source':'MCMRT article supplementary data',
                         'source_revision':manifest['sha256'],'source_url':manifest['article'],
                         'source_cell':f"3!{book['3'].cell(1,9+j).column_letter}{i}",
                         'license':manifest['license'],'run_id':None,'peak_width':None,'Rs':None})
    book.close();formulas.close()
    out.mkdir(parents=True,exist_ok=True)
    (out/'systems.json').write_text(json.dumps(systems,indent=2)+'\n')
    (out/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'missing_outcomes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in missing_outcomes))
    keys={'TFV':'SGOIRFVFHAKUTI-ZCFIWIBFSA-N','FTC':'XQSPYNMVSIKCOC-NTSWFWBYSA-N'}
    report={'observations':len(rows),'molecules':len({r['compound_id'] for r in rows}),'systems':len(systems),
            'column_products':len({s['metadata']['column.name'] for s in systems.values()}),
            'methods_B_methanol':sum(s['metadata']['eluent.B.meoh']==100 for s in systems.values()),
            'methods_B_acetonitrile':sum(s['metadata']['eluent.B.acn']==100 for s in systems.values()),
            'research_eligible_rows':sum(systems[r['system_id']]['eligible_for_rt_research'] for r in rows),
            'not_detected_outcomes_preserved':len(missing_outcomes),
            'issue_counts':dict(Counter(r['kind'] for r in issues)),'issues':issues,
            'target_observations':{name:sum(r['compound_id']==key for r in rows) for name,key in keys.items()},
            'source_manifest':manifest,'source_workbook':'Supplementary Data 1-14.xlsx',
            'note':'Only experimental sheet 3; author predictions/classifications/calibrant flags excluded. No gradient-derived k imported as isocratic k.'}
    (out/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/external/mcmrt'))
    p.add_argument('--out',type=Path,default=Path('data/processed/public_retention/mcmrt'))
    a=p.parse_args();r=build(a.root,a.out)
    print(json.dumps({k:v for k,v in r.items() if k not in ('issues','source_manifest')},indent=2))
