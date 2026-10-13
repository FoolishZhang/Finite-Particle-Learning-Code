"""Thin calls into the original production modules; all outputs are fresh.

Install at tools/production_replay.py in a clean export. Paper mode is expensive
and intentionally separate from manufactured small mode. No original run,
freeze, delivery, optimizer, or manuscript wrapper is invoked.
"""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import hashlib
import importlib
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
import unittest

TARGETS=('teacher','candidate_probability','gradient','hessian','root_kl_saved',
         'angular_pair','angular_audit','r02_active','r02_noise','r02_weights',
         'r03_teacher','r03_candidate','r03_risk','figure1_kernels','probability_audit','figure2')


def require(ok,message):
    if not ok:raise ValueError(message)


def primitive(value):
    if isinstance(value,dict):return {str(k):primitive(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [primitive(v) for v in value]
    if isinstance(value,F):return str(value)
    if hasattr(value,'tolist'):return primitive(value.tolist())
    if hasattr(value,'item'):return primitive(value.item())
    return value


class Runner:
    def __init__(self,source,output,mode):
        self.source=source.resolve();self.output=output.resolve();self.mode=mode;self.inputs={};self.loaded={}
        require(self.source.is_dir() and (self.source/'script').is_dir(),'Source export with script/ required')
        require(not self.output.is_relative_to(self.source),'Output must be outside source export')
        require(not self.output.exists(),'Output must be a new directory; no overwrite')
        self.output.mkdir(parents=True)
        temp=self.output/'temporary-fixtures';temp.mkdir();tempfile.tempdir=str(temp)
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(self.source/'script'))
        def guard(event,args):
            if event=='open' and isinstance(args[0],(str,bytes)):
                p=Path(args[0]).resolve();mode=args[1] if len(args)>1 else None
                flags=args[2] if len(args)>2 else 0
                writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (1|2|64|512|1024))
                if writing:require(p.is_relative_to(self.output),'Write outside new output: '+str(p))
                if p.suffix.lower() in ('.npz','.npy') and not p.is_relative_to(self.output):raise ValueError('No historical scientific arrays may be read')
            if event in ('subprocess.Popen','os.system'):raise ValueError('Producer wrapper/subprocess forbidden')
        sys.addaudithook(guard)

    def module(self,name):
        obj=importlib.import_module(name)
        p=Path(obj.__file__).resolve()
        require(p.is_relative_to(self.source/'script'),'Unexpected scientific module origin')
        self.loaded[p.relative_to(self.source).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return obj

    def read(self,pkg,name):
        p=self.source/'artifacts'/pkg/name
        require(p.is_relative_to(self.source) and p.is_file(),'Scientific input missing: '+str(p))
        self.inputs[p.relative_to(self.source).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf8'))

    def save(self,name,value):
        p=self.output/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf8',newline='\n') as f:json.dump(primitive(value),f,indent=2,allow_nan=False);f.write('\n')

    def tests(self,module,class_name=None):
        m=self.module(module)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(getattr(m,class_name)) if class_name else unittest.defaultTestLoader.loadTestsFromModule(m)
        log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        require(result.wasSuccessful(),log.getvalue())
        return dict(passed=True,original_manufactured_tests=result.testsRun,log=log.getvalue(),new_scientific_target_evaluations=0)

    def execute(self,target):
        if self.mode=='small':return self.small(target)
        return self.paper(target)

    def small(self,target):
        if target=='teacher':return self.tests('certified_teacher_compute_test','ManufacturedTests')
        if target=='candidate_probability':return self.tests('certified_teacher_risk_test','Manufactured')
        if target=='gradient':return self.tests('certified_f8_gradient_compute_test','ManufacturedTests')
        if target=='hessian':return self.tests('certified_f8_hessian_compute_test','ManufacturedTests')
        if target=='root_kl_saved':
            path=self.source/'supplementary/h2_2';sys.path.insert(0,str(path));check=importlib.import_module('check')
            require(Path(check.__file__).resolve()==path/'check.py','Wrong saved checker')
            return dict(passed=True,verification_layer='saved arithmetic, not production integration',result=check.theorem3())
        if target in ('angular_pair','angular_audit'):
            name='angular_witness_integrals' if target=='angular_pair' else 'angular_witness_path_audit';m=self.module(name)
            if target=='angular_audit':return m.self_test()
            a=tuple(F(i,16) for i in range(1,9));u=tuple(range(-7,8,2));r=m.finite_pair_sum(a,u,step='1/2',N=1,bits=192,hard_seconds=10)
            require(r['status']=='completed' and r['nodes_completed']==3,'Small pair finite sum')
            return dict(passed=True,manufactured_m=8,manufactured_nodes=3,finite_sum=r,not_a_true_integral_certificate=True)
        if target=='r02_active':
            m=self.module('r02_finite_scale_constants');r=m.compute_constants(step='1',N=2,strip='1/2',bits=128,hard_seconds=20)
            require(r['status']=='complete','Small active constant finite sum')
            return dict(passed=True,manufactured_grid=True,result=r,not_paper_error_certificate=True)
        if target=='r02_noise':
            m=self.module('r02_remainder_noise');from flint import arb,ctx
            ctx.prec=128;c={k:m.record(arb(v)) for k,v in (('v',1),('d0',0),('kappa3_L',0))}
            r=m.finite_noise(c,step='1',N=2,u_start='1/4',u_step='1/4',u_count=3,strip='1/2',bits=128,hard_seconds=20)
            require(r['status']=='complete','Small noise finite sum')
            return dict(passed=True,manufactured_input_constants=True,result=r,not_paper_error_certificate=True)
        if target=='r02_weights':
            m=self.module('r02_remainder_propagate');r=m.make_weight_grid(step='1',N=2,bits=128,hard_seconds=20)
            require(r['status']=='complete','Small comparison finite sum')
            return dict(passed=True,manufactured_grid=True,result=r)
        if target=='r03_teacher':return self.module('r03_teacher_derivative').self_test()
        if target=='r03_candidate':return self.module('r03_candidate_interval').self_test()
        if target=='r03_risk':
            m=self.module('r03_risk_interval');checks=m.self_test();u=list(range(-7,8,2));d=dict(u_integer=u,S_integer=sum(x*x for x in u))
            r=m.finite_risk(['1/2']*8,d,step='1/2',N=2,bits=128,hard_seconds=20)
            require(r['status']=='complete' and len(r['Q_finite'])==len(r['N_finite'])==256,'Small risk finite sum')
            return dict(passed=True,algebra=checks,manufactured_grid=True,result=r,not_a_positive_denominator_or_true_curvature_certificate=True)
        if target=='figure1_kernels':
            m=self.module('failure_probability_compute');import numpy as np
            z,s,common=m.manufactured_fixture(5);point=m.point_differences(z,s)
            q=np.ones(8)/8
            reference=m.legacy.score_pair(np.eye(3),z,s,q)[:,1]-m.legacy.score_pair(m.legacy.CB,z,s,q)[:,1]
            require(np.max(np.abs(point@q-reference))<=m.TOLERANCE,'Original label vs score kernels')
            audit=m.audit_single_bank(z[0],s[0],point[0]);require(all(r['passed'] for r in audit['rows']),'Original interval score audit')
            kernel=self.module('local_geometry_kernel');ordinary=kernel.price_curvature(z,s)
            intervals=kernel.interval_curvature(z[0],s[0]);errors=[]
            for ni,row in enumerate(intervals):
                for ji,value in enumerate(row):errors.append(float(abs(value-float(ordinary[0,ni,ji])).upper()))
            require(max(errors)<=kernel.QUALIFICATION_TOLERANCE,'Original curvature kernel vs Arb on manufactured bank')
            geometry=dict(passed=True,manufactured_bank=1,conditions=2,correlation_directions=3,
                          max_Arb_difference_upper=max(errors),tolerance=kernel.QUALIFICATION_TOLERANCE,
                          original_torch_autograd_qualification_not_run=True)
            return dict(passed=True,manufactured_banks=6,particles=5,label_differences=48,original_interval_audit=audit,geometry=geometry,new_rng_draws=0)
        if target=='probability_audit':
            m=self.module('failure_probability_audit');return self.probability(m,paper=False)
        if target=='figure2':
            qualification=self.module('asymptotic_bridge_core_test').qualification()
            worker=self.module('asymptotic_bridge_run');out=self.output/'manufactured_bridge';out.mkdir();(out/'raw').mkdir()
            self.save('manufactured_bridge/REFERENCE.json',dict(m_H45=.003,manufactured_only=True))
            self.save('manufactured_bridge/FREEZE.json',dict(schema='manufactured_scheduler_fixture_v1',seed_family=202610130902,banks=1,particles=2))
            job=dict(out=str(out),block_id=0,seed_family=202610130902,banks=1,particles=2,m_H=.003,
                     freeze_sha=hashlib.sha256((out/'FREEZE.json').read_bytes()).hexdigest(),
                     reference_sha=hashlib.sha256((out/'REFERENCE.json').read_bytes()).hexdigest(),scientific=False,attempt=1)
            receipt=worker.worker_block(job)
            require(receipt['banks']==1 and receipt['metadata']['scientific'] is False,'Manufactured original worker receipt')
            return dict(passed=True,original_core_qualification=qualification,original_worker_receipt=receipt,
                        scheduler_fixture_Gaussian_scalars=2*4099,scheduler_fixture_uniform_draws=1,
                        scientific_sampling=False)
        raise ValueError(target)

    def paper(self,target):
        """Regenerate declared paper-scale finite objects; not run during release."""
        from flint import ctx
        ctx.prec=192
        if target=='root_kl_saved':return self.small(target)
        if target=='teacher':
            m=self.module('certified_teacher_compute');inp=self.read('certified_teacher_v1_20261008','INPUTS.json');t=inp['exact_teacher'];g=inp['grid'];rows=list(range(g['index_min'],g['index_max']+1));blocks=[];deadline=time.perf_counter()+inp['hard_wall_budget_seconds']
            for start in range(0,len(rows),g['outer_rows_per_block']):
                b=m.finite_grid_block(t['angles'],t['row_length'],g['h'],g['index_min'],g['index_max'],rows[start:start+g['outer_rows_per_block']],precision_bits=inp['precision_bits'],label_mode=inp['label_mode'],deadline=deadline)
                self.save(f'block_{start:04}.json',b);blocks.append(b)
            return m.aggregate_blocks(blocks,t['observations'],inp['label_mode'])
        if target=='candidate_probability':
            m=self.module('certified_teacher_risk');inp=self.read('certified_teacher_v1_20261008','CANDIDATE_RISK_INTEGRALS.json')
            decode=lambda xs:[F(int(x['numerator']),int(x['denominator'])) for x in xs]
            return m.finite_candidate(decode(inp['a_exact']),decode(inp['h_exact']))
        if target in ('gradient','hessian'):
            prefix='certified_f8_'+target;m=self.module(prefix+'_compute');inp=self.read(prefix+'_v1_20261008','INPUTS.json');deadline=time.perf_counter()+inp['hard_total_wall_seconds'];cache=m.precompute_inner(inp['a_pairs_exact'],inp['inner_grid']['h_exact'],inp['inner_grid']['L_exact']);o=inp['outer_grid'];rows=[]
            reps=inp['mapping']['representatives'] if target=='gradient' else inp['teacher']['representatives']
            for i,c in enumerate(reps):
                if target=='gradient':r=m.finite_representative(cache,c,o['h_exact'],o['vmin_exact'],o['vmax_exact'],K=inp['K'],backend=inp['backend'],saturation_threshold=inp['saturation_threshold'],deadline=deadline)
                else:
                    W,M,S=m.restore_moment(inp['teacher']['teacher_moments'][i]);r=m.finite_representative(cache,c,W,M,S,o['h_exact'],o['vmin_exact'],o['vmax_exact'],K=inp['K'],saturation_threshold=inp['saturation_threshold'],deadline=deadline)
                self.save(f'rep_{i:02}.json',r);rows.append(r)
            return dict(completed=True,representatives=len(rows),finite_matrix_blocks=m.aggregate_records(rows) if target=='hessian' else None,analytic_errors_not_added=True)
        if target in ('angular_pair','angular_audit'):
            m=self.module('angular_witness_integrals' if target=='angular_pair' else 'angular_witness_path_audit');inp=self.read('angular_witness_v1_20261010','INPUTS.json');fn=m.finite_pair_sum if target=='angular_pair' else m.finite_path_audit
            return fn(inp['a_exact'],inp['u_integer'],step=inp['step_exact'],N=inp['N'],bits=inp['bits'],hard_seconds=inp['hard_seconds'])
        if target=='r02_active':
            m=self.module('r02_finite_scale_constants');inp=self.read('r02_finite_scale_v1_20261010','INPUTS.json')
            return m.compute_constants(**inp['parameters'])
        if target=='r02_noise':
            m=self.module('r02_remainder_noise');inp=self.read('r02_remainder_v1_20261010','INPUTS.json');old=self.read('r02_finite_scale_v1_20261010','CONSTANTS.json')['composition']['constants']
            return m.finite_noise(old,**inp['parameters'])
        if target=='r02_weights':
            m=self.module('r02_remainder_propagate');inp=self.read('r02_remainder_v1_20261010','INPUTS.json')
            return m.make_weight_grid(**inp['weight_parameters'])
        if target.startswith('r03_'):
            inp=self.read('r03_teacher_family_v1_20261011','BOX_INPUTS.json')
            if target=='r03_teacher':
                m=self.module('r03_teacher_derivative');block=m.finite_grid_block('0',config=inp['teacher_config'],deadline=time.perf_counter()+inp['teacher_seconds']);self.save('finite_block.json',block)
                finite=m.aggregate_blocks([block]);return dict(finite=finite,composition=m.compose(finite))
            m=self.module('r03_candidate_interval' if target=='r03_candidate' else 'r03_risk_interval');output={}
            for key in inp['box_order']:
                box=inp['boxes'][key]
                if target=='r03_risk':r=m.finite_risk(box['a_input_boxes'],inp['direction'],step=inp['risk_step'],N=inp['risk_N'],bits=192,hard_seconds=inp['risk_seconds_per_box'])
                else:
                    cache=m.precompute(box['a_input_boxes'],inp['candidate_config']);rows=[];deadline=time.perf_counter()+inp['candidate_seconds_per_box']
                    for first in range(0,128,8):
                        b=m.finite_label_block(cache,range(first,first+8),deadline=deadline);self.save(key+f'/LABELS_{first:03}.json',b);rows.append(b)
                    r=dict(completed=True,blocks=len(rows),analytic_bounds=m.analytic_error_bounds(box['a_input_boxes'],inp['candidate_config'],teacher_error_L1_upper='0',teacher_mass_upper='1'))
                self.save(key+'/result.json',r);output[key]=dict(completed=r.get('status')=='complete' or r.get('completed') is True)
            return output
        if target=='probability_audit':return self.probability(self.module('failure_probability_audit'),paper=True)
        if target=='figure2':
            core=self.module('asymptotic_bridge_core');run=self.module('asymptotic_bridge_run');ref=self.read('asymptotic_bridge_v1_20261008','REFERENCE.json');cfg=core.production_config();out=self.output/'blocks';out.mkdir();(out/'raw').mkdir()
            self.save('blocks/REFERENCE.json',ref)
            self.save('blocks/FREEZE.json',dict(schema='fresh_export_scientific_configuration_v1',
                      config=cfg,source_modules=dict(self.loaded),source_inputs=dict(self.inputs),
                      historical_author_freeze_not_reused=True,new_output_only=True))
            freeze_sha=hashlib.sha256((out/'FREEZE.json').read_bytes()).hexdigest()
            reference_sha=hashlib.sha256((out/'REFERENCE.json').read_bytes()).hexdigest()
            records=[];import numpy as np
            scores=np.empty((cfg['banks'],len(core.N_VALUES),len(core.METRIC_NAMES)),dtype=np.float64)
            for i in range(cfg['block_count']):
                job=dict(out=str(out),block_id=i,seed_family=cfg['seed_family'],banks=cfg['block'],particles=cfg['particles'],m_H=ref['m_H45'],freeze_sha=freeze_sha,reference_sha=reference_sha,scientific=True,attempt=1)
                receipt=run.worker_block(job);records.append(receipt)
                with np.load(out/receipt['path'],allow_pickle=False) as saved:
                    scores[i*cfg['block']:(i+1)*cfg['block']]=saved['scores']
            np.save(self.output/'BANK_SCORES.npy',scores)
            summary=self.module('asymptotic_bridge_statistics').analyse(scores,ref)
            self.save('STATISTICS.json',summary)
            return dict(completed=True,blocks=len(records),new_scientific_sampling=True,
                        fresh_conditional_statistics_reestimated=True,reference_coefficient_reused=True,records=records)
        if target=='figure1_kernels':raise ValueError('Figure 1 full historical selection requires the original training-count asset and freeze wrapper; only bounded scoring/curvature production is adapted here. See REPRODUCING.')
        raise ValueError(target)

    def probability(self,m,paper):
        import numpy as np
        # Exactly the original bank generator/scoring loop, with a distinct
        # fixture seed and small K/count in the release test only.
        count,K,block=(m.L,m.K,m.BLOCK) if paper else (4,5,2)
        seed=m.SEED if paper else 202610130901
        zr=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([seed,1])))
        sr=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence([seed,2])))
        differences=[];truncated=[];Z=[];S=[];tick=time.monotonic()
        for start in range(0,count,block):
            n=min(block,count-start);z=zr.standard_normal((n,K,3));s=-np.logaddexp(0,-sr.standard_normal((n,K,256))).sum(axis=2)
            d=m.development.point_differences(z,s);differences.append(d);truncated.append(np.max(np.abs(z),axis=(1,2))>m.T);Z.append(z);S.append(s)
            require(time.monotonic()-tick<m.MAX_SECONDS,'Original audit production budget exhausted')
        D=np.concatenate(differences);outside=np.concatenate(truncated)
        require(D.shape==(count,8) and np.isfinite(D).all(),'Generated eight-label audit differences')
        x=np.where(outside,0.,D.mean(axis=1));delta=np.where(outside,0.,D[:,[0,7]].mean(axis=1)-D[:,1:7].mean(axis=1))
        from flint import ctx
        ctx.prec=128;moments={name:m.exact_stats(v)[0] for name,v in (('X',x),('X_squared',x*x),('D_T',delta))}
        if paper:
            np.savez(self.output/'AUDIT_BANKS.npz',z=np.concatenate(Z),s=np.concatenate(S),difference_by_label=D,truncated=outside,label_order=m.development.YS,X=x,X_squared=x*x,D_T=delta)
            inference=m.infer(moments)
            c=m.constants()
            from flint import arb
            range_gates=dict(X=all(abs(arb(float(v)))<=c['M']+c['epsilon_X'] for v in (x.min(),x.max())),
                             X_squared=bool(arb(float((x*x).min()))>=0 and arb(float((x*x).max()))<=c['M']**2+c['epsilon_X_squared']),
                             D_T=all(abs(arb(float(v)))<=2*c['M']+c['epsilon_D'] for v in (delta.min(),delta.max())))
            require(all(range_gates.values()),'Original audit range qualifications')
        else:inference=None
        return dict(passed=True,completed=True,paper_scale=paper,banks=count,particles=K,
                    generated_Gaussian_scalars=count*K*259,seed_family=seed,scoring_function='failure_probability_compute.point_differences',
                    exact_binary64_moments=moments,inference=inference,
                    conditional_population_inference_returned=paper,ideal_iid_and_numerical_allowances_remain_conditions=True,manufactured_rng_only=not paper,
                    scientific_reference_rewritten=False,full_score_error_replay_not_executed=True)

    def run(self,targets):
        tick=time.monotonic();results={}
        for target in targets:
            start=time.monotonic()
            try:
                result=self.execute(target);self.save(target+'/result.json',result)
                if self.mode=='paper' and isinstance(result,dict):
                    require(result.get('status') not in ('budget_stop','budget_stopped') and result.get('completed') is not False,'Paper-grid producer stopped; partial output retained')
                    require(not any(isinstance(v,dict) and v.get('completed') is False for v in result.values()),'Paper-grid box incomplete; original partial outputs retained')
                results[target]=dict(passed=True,wall_seconds=round(time.monotonic()-start,3),output=target+'/result.json')
            except BaseException as error:
                partial=getattr(error,'partial',None)
                self.save(target+'/failure.json',dict(type=type(error).__name__,message=str(error),partial=partial))
                raise
            print(json.dumps(dict(target=target,**results[target])),flush=True)
        loaded={}
        for obj in list(sys.modules.values()):
            p=getattr(obj,'__file__',None)
            if p and Path(p).resolve().is_relative_to(self.source/'script'):
                path=Path(p).resolve();loaded[path.relative_to(self.source).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        report=dict(passed=True,mode=self.mode,results=results,python=platform.python_version(),platform=platform.platform(),
                    source_inputs=self.inputs,original_loaded_modules=loaded,total_wall_seconds=round(time.monotonic()-tick,3),
                    paper_scale_executed=self.mode=='paper',new_science_claims=False,
                    original_scientific_reference_files_rewritten=False,scientific_arrays_read=False,
                    scope='bounded manufactured production tests' if self.mode=='small' else 'fresh paper-grid production objects; interpretation and analytic qualifications remain separate')
        self.save('RUN.json',report);print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--mode',choices=('small','paper'),default='small');parser.add_argument('--target',choices=(*TARGETS,'all'),default='all')
    args=parser.parse_args();require(sys.flags.optimize==0,'Keep scientific assertions enabled')
    require(not(args.mode=='paper' and args.target=='all'),'Select one paper-scale target explicitly')
    targets=TARGETS if args.target=='all' else (args.target,)
    Runner(args.source_root,args.output_root,args.mode).run(targets)


if __name__=='__main__':main()
