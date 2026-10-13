"""Derive a compact report from frozen evidence; never rerun science."""
from pathlib import Path
import argparse, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nuisance_dimensions'
DOC=ROOT/'docs/08_research_研究工作/无关维度与固定噪声总体偏好_20261001'
def read(p): return json.loads(Path(p).read_text('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def derive():
    mc=read(OUT/'MC/RESULTS.json'); num=read(OUT/'NUMERIC/RESULTS.json')
    checks=read(OUT/'NUMERIC/CHECKS.json')['rows']
    return dict(source_sha256=sha(__file__),
      inputs={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
              [OUT/'MC/RESULTS.json',OUT/'NUMERIC/RESULTS.json',OUT/'NUMERIC/CHECKS.json']},
      mc_main_pass=mc['main_passed'],mc_main_total=mc['main_total'],mc_quantities=len(mc['records']),
      mc_gaps=[x for x in mc['records'] if x['kind']=='gap_R_minus_0'],
      mc_finite_risk_certificates=mc['finite_n_risk_certificates'],mc_ledger=mc['ledger'],
      numeric_qualification_count=len(checks),numeric_qualification_pass=sum(x['passed'] for x in checks),
      numeric_prediction_pass=num['prediction_passed'],numeric_prediction_total=num['prediction_total'],
      numeric_predictions=num['prediction_readouts'],numeric_ledger=num['ledger'],
      native_CF_integrals=sum(x['native_evaluations'] for x in num['ledger']['engines'].values()),
      nuisance_CF_integrals=sum(x['nuisance_evaluations'] for x in num['ledger']['engines'].values()),
      native_phase_terms=sum(x['native_phase_node_terms'] for x in num['ledger']['engines'].values()),
      nuisance_phase_terms=sum(x['nuisance_phase_node_terms'] for x in num['ledger']['engines'].values()),
      numerical_qualification_not_full_integral_certificate=True, new_scientific_computation=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['write','check']);a=p.parse_args()
    result=derive();path=OUT/'DERIVED_SUMMARY.json'
    if a.action=='write':
        with path.open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    else: assert read(path)==result
    print(json.dumps({k:result[k] for k in ['mc_main_pass','mc_main_total','numeric_qualification_count',
       'numeric_qualification_pass','numeric_prediction_pass','numeric_prediction_total','native_CF_integrals',
       'nuisance_CF_integrals','native_phase_terms','nuisance_phase_terms','new_scientific_computation']}))
if __name__=='__main__':main()
