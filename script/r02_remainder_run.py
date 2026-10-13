"""Frozen finite-remainder execution, without direct D_n scoring."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import argparse,hashlib,json,platform,sys,traceback
import r02_remainder_noise as noise
import r02_remainder_audit as audit
import r02_remainder_propagate as propagation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/r02_remainder_v1_20261010'
DOC='docs/08_research_研究工作/R02Remainder_v1_20261010/'
BASE='31399da0ae8aa38333c0d6dd97ec03ac58937556'
FEEDBACK='86b11895ba9fef8311cafe03a302b760ce5ac073c1774b0760cd40388c2c0611'
OLD='artifacts/r02_finite_scale_v1_20261010/CONSTANTS.json'
PARENT='artifacts/r02_finite_scale_v1_20261010/DELIVERY_MANIFEST.json'
PARENT_SHA='e37a8655e13722bd4a33ad822fed0998803735d2684071a23705940d2fd52e83'
PARAMS=dict(step='1/16',N=192,u_start='1/4',u_step='1/64',u_count=273,strip='1',bits=192,hard_seconds=600)
WEIGHT_PARAMS=dict(step='1/128',N=1536,bits=192,hard_seconds=120)
SOURCES=[OLD,PARENT,'artifacts/r02_finite_scale_v1_20261010/RESULTS.json',
 'artifacts/submission_candidate_v1_9_20261010/DELIVERY_MANIFEST.json',
 'script/r02_remainder_run.py','script/r02_remainder_noise.py','script/r02_remainder_audit.py','script/r02_remainder_propagate.py',
 'artifacts/r02_remainder_v1_20261010/NOISE_IMPLEMENTATION_CHECK.json']+[DOC+x for x in
 ('CONTRACT.md','FOURIER_PLAN.md','NOISE_CERTIFICATE_METHOD.md','WEIGHTED_INTERFACE.md','FINITE_PROPAGATION.md','WEIGHT_CERTIFICATE_METHOD.md','AUDIT_METHOD.md')]

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def now():return datetime.now(timezone.utc).isoformat()
def write_new(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lo(r):return noise.endpoint_fraction(r['lower_exact_dyadic'])
def hi(r):return noise.endpoint_fraction(r['upper_exact_dyadic'])
def float_fraction(x):return F.from_float(x) if isinstance(x,float) else F(x)

def freeze():
    require(sha(ROOT/'HANDOFF.md')==FEEDBACK,'Unexpected feedback change')
    require(sha(ROOT/PARENT)==PARENT_SHA,'Parent identity')
    require(not (OUT/'FREEZE.json').exists(),'Freeze once')
    with (OUT/'HANDOFF_INPUT.md').open('xb') as f:f.write((ROOT/'HANDOFF.md').read_bytes())
    inputs=dict(schema='r02_remainder_inputs_v1',baseline=BASE,
        current_user_authorization='继续推进R02到下一个合理终点；继续推进R02；继续',
        handoff_sha256=FEEDBACK,handoff_updated_this_round=False,feedback_reviewed_commit='513f15a8c6fc3d2f74db4c770a2f377dd290c922',
        parameters=PARAMS,weight_parameters=WEIGHT_PARAMS,audit_hard_seconds=120,
        audit_CF_gate_exact='1e-10',audit_mu4_gate_exact='1e-9',n_values=[64,256,1024,4096],K=100,rho_exact='9/20',
        model_and_coordinate='m3 independent centered teacher, free c12 at equicorrelation .45',
        direct_D_n_scoring=False,old_active_integrals_reused=True,sign_not_a_qualification_gate=True)
    write_new(OUT/'INPUTS.json',inputs)
    paths=SOURCES+[(OUT/x).relative_to(ROOT).as_posix() for x in ('INPUTS.json','HANDOFF_INPUT.md')]
    write_new(OUT/'FREEZE.json',dict(schema='r02_remainder_freeze_v1',baseline=BASE,frozen_utc=now(),
        bindings={x:sha(ROOT/x) for x in paths},new_production_noise_nodes_before_freeze=0,
        new_production_weight_nodes_before_freeze=0,new_audit_target_nodes_before_freeze=0))

def verified_inputs():
    lock=read(OUT/'FREEZE.json')
    for name,digest in lock['bindings'].items():require(sha(ROOT/name)==digest,'Frozen source changed: '+name)
    inp=read(OUT/'INPUTS.json')
    require(inp['parameters']==PARAMS and inp['weight_parameters']==WEIGHT_PARAMS and inp['n_values']==[64,256,1024,4096],'Fixed inputs')
    return inp,read(ROOT/OLD)['composition']['constants']

def run(kind):
    inp,old=verified_inputs();marker=OUT/(kind+'_START.json')
    write_new(marker,dict(schema='r02_remainder_start_v1',kind=kind,started_utc=now(),
        freeze_sha256=sha(OUT/'FREEZE.json'),input_sha256=sha(OUT/'INPUTS.json'),
        executable=sys.executable,python=platform.python_version(),attempt=1,automatic_retry=False))
    try:
        if kind=='NOISE':r=noise.finite_noise(old,**PARAMS)
        elif kind=='WEIGHTS':r=propagation.make_weight_grid(**WEIGHT_PARAMS)
        else:r=audit.compute_audit(old,hard_seconds=inp['audit_hard_seconds'])
        r['run_provenance']=dict(marker_sha256=sha(marker),freeze_sha256=sha(OUT/'FREEZE.json'),input_sha256=sha(OUT/'INPUTS.json'),finished_utc=now())
        write_new(OUT/(kind+'.json'),r)
        require(r['status']=='complete',kind+' incomplete; retained original result')
        if kind=='NOISE':require(all(r['composition']['checks'].values()),'Noise input qualification failed')
        print(json.dumps(dict(kind=kind,status=r['status'],counts=r.get('work_counts',r.get('counts')),seconds=r.get('elapsed_seconds')),indent=2))
    except BaseException:
        write_new(OUT/(kind+'_FAILURE.json'),dict(failed_utc=now(),traceback=traceback.format_exc(),automatic_retry=False));raise

def audit_comparison(primary,other,inp):
    require(other['status']=='complete','Independent audit incomplete')
    cfg=other['configuration']
    require(cfg['GL_orders']==[384,512] and cfg['window']==[-12,12] and not cfg['quadrature_weight_renormalization'],'Audit rule')
    require(cfg['u_start_exact']=='1/4' and cfg['u_step_exact']=='1/64' and cfg['u_count']==273,'Audit frequencies')
    require(other['elapsed_seconds']<=120,'Audit budget')
    rows=[];a,b=other['by_order']['384'],other['by_order']['512']
    for k,row in enumerate(primary['cf_rows']):
        for part in ('real','imag'):
            ref=row['true_'+part];x,y=float_fraction(a['CF_'+part][k]),float_fraction(b['CF_'+part][k])
            d=[max(F(0),lo(ref)-v,v-hi(ref)) for v in (x,y)]
            gate=F(inp['audit_CF_gate_exact'])
            rows.append(dict(index=k,u_exact=row['u_exact'],part=part,distance384_exact=str(d[0]),distance512_exact=str(d[1]),
                rule_difference_exact=str(abs(x-y)),passed=max(d)<gate and abs(x-y)<gate))
    ref=primary['composition']['mu4_true'];x,y=float_fraction(a['mu4']),float_fraction(b['mu4'])
    dist=[max(F(0),lo(ref)-v,v-hi(ref)) for v in (x,y)];gate=F(inp['audit_mu4_gate_exact'])
    moment=dict(distance384_exact=str(dist[0]),distance512_exact=str(dist[1]),rule_difference_exact=str(abs(x-y)),passed=max(dist)<gate and abs(x-y)<gate)
    require(all(r['passed'] for r in rows) and moment['passed'],'Independent expression qualification failed')
    return dict(CF_components=rows,mu4=moment,passed=True,qualification='ordinary floating comparison, not a second rigorous certificate')

def analyze():
    inp,old=verified_inputs();data={x:read(OUT/(x+'.json')) for x in ('NOISE','WEIGHTS','AUDIT')}
    for kind,r in data.items():
        require(r['status']=='complete',kind+' not complete');p=r['run_provenance']
        require(p['marker_sha256']==sha(OUT/(kind+'_START.json')) and p['freeze_sha256']==sha(OUT/'FREEZE.json') and p['input_sha256']==sha(OUT/'INPUTS.json'),kind+' run binding')
    raw=data['NOISE'];cfg=raw['configuration']
    require(cfg['N']==192 and cfg['precision_bits']==192 and cfg['u_start_exact']=='1/4' and cfg['u_end_exact']=='9/2' and cfg['u_count']==273,'Noise production parameters')
    require(raw['elapsed_seconds']<=600 and raw['work_counts']['new_complex_exp_nodes']==105105 and raw['work_counts']['new_gaussian_nodes']==385,'Noise execution count/budget')
    require(raw['input_constants']=={k:old[k] for k in raw['input_constants']},'Old true inputs')
    wg=data['WEIGHTS'];wc=wg['configuration']
    require(wc['step_exact']=='1/128' and wc['N']==1536 and wc['limit_exact']=='12' and wc['precision_bits']==192,'Weight production parameters')
    require(wg['nodes_completed']==3073 and wg['work_counts']['new_weight_grid_nodes']==3073 and wg['elapsed_seconds']<=120,'Weight execution count/budget')
    base=noise.compose_saved(raw);require(base==raw['composition'] and all(base['checks'].values()),'Noise saved composition')
    comparisons=audit_comparison(raw,data['AUDIT'],inp)
    bounds=noise.compose_bounds(raw,inp['n_values'])
    result=propagation.propagate(old,bounds,data['WEIGHTS'])
    return dict(schema='r02_remainder_results_v1',baseline=BASE,
        freeze_sha256=sha(OUT/'FREEZE.json'),noise_sha256=sha(OUT/'NOISE.json'),weights_sha256=sha(OUT/'WEIGHTS.json'),audit_sha256=sha(OUT/'AUDIT.json'),
        noise_bounds=bounds,independent_expression_check=comparisons,propagation=result,
        no_direct_D_n_scoring=True,new_D_n_evaluations=0,new_random_samples=0,new_NPZ_reads=0,
        new_production_integrals_during_composition=0,manuscript_changed=False,R03_executed=False)

def main():
    require(sys.flags.optimize==0,'Assertions must be enabled')
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for x in ('freeze','run-noise','run-weights','run-audit','compose','check'):g.add_argument('--'+x,action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.run_noise or a.run_weights or a.run_audit:run('NOISE' if a.run_noise else 'WEIGHTS' if a.run_weights else 'AUDIT')
    else:
        r=analyze()
        if a.compose:write_new(OUT/'RESULTS.json',r)
        else:require(r==read(OUT/'RESULTS.json'),'Saved result differs')
        print(json.dumps(dict(passed=True,mode='compose' if a.compose else 'check',
            propagation_keys=list(r['propagation']),new_production_integrals=0,no_direct_D_n_scoring=True),indent=2))

if __name__=='__main__':main()
