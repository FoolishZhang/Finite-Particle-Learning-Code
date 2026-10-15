"""Saved fixed-cell local-shrinkage summaries; no sampling or model evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import numpy as np
from scipy.stats import t

METHODS=('integrated','group8','ridge')
BASELINES=('I','original_two_start_baseline','original_other_initial')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,default=serial,allow_nan=False);f.write('\n')
def csv_write(path,rows):
    if not rows:raise ValueError('nonempty saved table required')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,default=serial) if isinstance(v,(dict,list,tuple,np.ndarray)) else v for k,v in row.items()})
def require(value,message):
    if not value:raise ValueError(message)


def load(attempt):
    attempt=Path(attempt).resolve();records={r['path']:r for r in read(attempt/'MANIFEST.json')['files']}
    inputs={}
    for name in ('SUMMARY.json','CONTRACT.json','FROZEN_OUTPUTS.json'):
        require(name in records and sha(attempt/name)==records[name]['sha256'],'producer identity '+name)
        inputs[name]={'path':str(attempt/name),'sha256':records[name]['sha256']}
    inputs['MANIFEST.json']={'path':str(attempt/'MANIFEST.json'),'sha256':sha(attempt/'MANIFEST.json')}
    report,contract=read(attempt/'SUMMARY.json'),read(attempt/'CONTRACT.json')
    require(report['schema']=='local_shrinkage_learning_v1' and report['status']=='completed','completed producer required')
    require(read(attempt/'FROZEN_OUTPUTS.json')['outputs']==report['outputs'],'all frozen outputs retained')
    return report,contract,inputs


def checked_cases(report,contract):
    cases=contract['cases'];by_id={c['case_id']:c for c in cases}
    require(len(by_id)==len(cases)==contract['case_count'],'unique explicit case menu')
    require(report['cases']==cases,'unchanged case order/menu')
    require(len(report['outputs'])==6*len(cases)==contract['main_outputs'],'all six fitted outputs per case')
    require(len(report['evaluations'])==9*len(cases)==contract['expected_risk_points'],'all nine risk rows per case')
    expected={(m,i) for m in METHODS for i in (0,1)}|{('initial',0),('initial',1),('teacher',None)}
    grouped={cid:{} for cid in by_id}
    for row in report['evaluations']:
        cid=row['case_id'];require(cid in by_id,'known explicit case')
        case=by_id[cid];key=(row['method'],row['initialization_index'])
        require(key in expected and key not in grouped[cid],'unique method/start risk')
        for k in ('cell_id','replicate','N'):require(row[k]==case[k],'case identity '+k)
        require(np.isfinite(row['joint_kl']) and isinstance(row['precision_qualified'],bool),'finite raw risk and boolean gate')
        if row['method']=='initial':require(row['coordinates']==contract['initial_coordinates'][row['initialization_index']],'matching initial')
        if row['method']=='teacher':require(row['coordinates']==case['coordinates'],'matching fixed teacher')
        grouped[cid][key]=row
    require(all(set(rows)==expected for rows in grouped.values()),'no missing or deleted rows')
    output_keys={(r['case_id'],r['method'],r['initialization_index']) for r in report['outputs']}
    require(len(output_keys)==6*len(cases) and output_keys=={(cid,m,i) for cid in by_id for m in METHODS for i in (0,1)},'all original fitted starts')
    return grouped


def numerical_rows(report,contract):
    grouped=checked_cases(report,contract);fit=[];risk=[];q=contract['qualification']
    for case in contract['cases']:
        for row in grouped[case['case_id']].values():
            N=case['N'];d=row['precision_diagnostics'];tol=min(q['kl_tol'],q['scaled_risk_tol']/N)
            valid=d['KL_order_difference']<=tol and d['PMF_order_max_difference']<=q['pmf_tol']
            require(valid==row['precision_qualified'],'saved risk gate arithmetic')
            risk.append({'case_id':case['case_id'],'cell_id':case['cell_id'],'replicate':case['replicate'],'N':N,
                'method':row['method'],'initialization_index':row['initialization_index'],'risk_qualified':valid,
                'KL_order_difference':d['KL_order_difference'],'N_times_KL_order_difference':N*d['KL_order_difference'],
                'PMF_order_max_difference':d['PMF_order_max_difference'],'risk_kl_tolerance':tol,'risk_pmf_tolerance':q['pmf_tol']})
            if row['method'] not in METHODS:continue
            own=row['own_context'];se=own.get('gradient_scramble_SE',[0.,0.,0.])
            fit.append({'case_id':case['case_id'],'cell_id':case['cell_id'],'replicate':case['replicate'],'N':N,
                'method':row['method'],'initialization_index':row['initialization_index'],'fit_status':row['fit_status'],
                'fit_mapping':row['fit_mapping_norm_inf'],'independent_mapping':own['projected_mapping']['norm_inf'],
                'max_gradient_SE':float(np.max(se)),'reference_diagnostic_passed':row['reference_qualified'],
                'NLL_order_difference':own.get('value_order_difference'),
                'gradient_order_max_difference':own.get('gradient_order_max_difference')})
    return grouped,fit,risk


def stats(values):
    values=np.asarray(values,float);require(values.ndim==1 and len(values)>1 and np.isfinite(values).all(),'independent finite case vector')
    return {'mean':float(values.mean()),'SE':float(values.std(ddof=1)/np.sqrt(len(values))),'case_values':values,'n':len(values)}


def case_values(grouped,contract):
    values=[]
    for case in contract['cases']:
        rows=grouped[case['case_id']];N=case['N']
        for method in (*METHODS,*BASELINES):
            if method in METHODS:chosen=[rows[(method,0)],rows[(method,1)]]
            elif method=='I':chosen=[rows[('initial',0)]]
            elif method=='original_other_initial':chosen=[rows[('initial',1)]]
            else:chosen=[rows[('initial',0)],rows[('initial',1)]]
            risks=[r['joint_kl'] for r in chosen];value=float(np.mean(risks))
            values.append({'case_id':case['case_id'],'cell_id':case['cell_id'],'replicate':case['replicate'],
                'N':N,'S':case['S'],'method':method,'start_risks':risks,'risk':value,'scaled_risk':N*value,
                'risk_qualified':all(r['precision_qualified'] for r in chosen)})
    return values


def validation_statistics(report,contract):
    require(report['phase']==contract['phase']=='validation','new fixed-cell validation required')
    grouped,fit,risk_precision=numerical_rows(report,contract)
    infer=contract['primary_inference'];n=infer['repetitions_per_cell'];cells=contract['cells']
    require(n==24 and len(cells)==4 and len(contract['cases'])==96,'four cells and 24 cases each')
    require(infer['two_sided'] is True and infer['family_size']==len(infer['comparisons'])==4,'four explicit two-sided comparisons')
    rows=case_values(grouped,contract);descriptive={};paired={};by_cell={}
    for cell in cells:
        cid=cell['cell_id'];cases=sorted([c for c in contract['cases'] if c['cell_id']==cid],key=lambda c:c['replicate'])
        require(len(cases)==n and [c['replicate'] for c in cases]==list(range(n)),'all independent dataset replicates')
        require(all(c['N']==cell['N'] and c['S']==cell['S'] for c in cases),'fixed cell N/S')
        selected={m:sorted([r for r in rows if r['cell_id']==cid and r['method']==m],key=lambda r:r['replicate']) for m in (*METHODS,*BASELINES)}
        by_cell[cid]=selected;descriptive[cid]={'N':cell['N'],'S':cell['S'],'case_ids':[c['case_id'] for c in cases],'methods':{}}
        for method,items in selected.items():
            descriptive[cid]['methods'][method]={'raw':stats([r['risk'] for r in items]),'scaled':stats([r['scaled_risk'] for r in items]),
                'unqualified_case_ids':[r['case_id'] for r in items if not r['risk_qualified']]}
        delta=np.array([r['risk'] for r in selected['group8']])-np.array([r['risk'] for r in selected['ridge']])
        paired[cid]={'raw':stats(delta),'scaled':stats(cell['N']*delta),'scope':'descriptive_only_not_equivalence_test'}
    critical=float(t.ppf(1-infer['alpha']/(2*infer['family_size']),n-1));primary=[]
    for contrast in infer['comparisons']:
        cid=contrast['cell_id'];left=by_cell[cid][contrast['left']];right=by_cell[cid][contrast['right']]
        delta=np.array([r['risk'] for r in left])-np.array([r['risk'] for r in right]);N=descriptive[cid]['N']
        raw,scaled=stats(delta),stats(N*delta)
        for unit in (raw,scaled):unit['CI']=[unit['mean']-critical*unit['SE'],unit['mean']+critical*unit['SE']]
        good=[a['risk_qualified'] and b['risk_qualified'] for a,b in zip(left,right)]
        direction='negative' if raw['CI'][1]<0 else 'positive' if raw['CI'][0]>0 else 'unresolved'
        primary.append({**contrast,'N':N,'case_ids':[r['case_id'] for r in left],'raw':raw,'scaled':scaled,
            'n':n,'df':n-1,'family_size':infer['family_size'],'alpha':infer['alpha'],'critical':critical,
            'all_risks_qualified':all(good),'unqualified_case_ids':[r['case_id'] for r,g in zip(left,good) if not g],
            'direction_supported':direction if all(good) else 'numerically_unqualified',
            'negative_realized_cases':int(np.sum(delta<0)),'positive_realized_cases':int(np.sum(delta>0))})
    return {'descriptive':descriptive,'group_minus_ridge_descriptive':paired,'primary_family':primary,
        'unit':'independent dataset within fixed cell; equal mean of two start risks, not risk at mean coordinates',
        'reference_diagnostic_pass_count':sum(r['reference_qualified'] is True for r in report['evaluations']),
        'reference_is_not_risk_inclusion_gate':True,'production_budget':report['budget'],'theory_coefficients':contract['theory_coefficients'],
        'qualification':'fixed_algorithm_finite_N_risk; local_limit_curves_are_not_finite_N_guarantees; no_teacher_population_or_equivalence_claim'},rows,fit,risk_precision


def begin(attempt,out):
    report,contract,inputs=load(attempt);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'SOURCE.py')
    inputs['analysis_source']={'path':str(out/'SOURCE.py'),'sha256':sha(out/'SOURCE.py')}
    write(out/'INPUT_HASHES.json',inputs);write(out/'FREEZE.json',{'before_arithmetic':True,'input_hashes':inputs,'scientific_RNG_draws':0,'new_model_calls':0})
    return report,contract,out


def seal(out):
    for entry in read(out/'INPUT_HASHES.json').values():require(sha(entry['path'])==entry['sha256'],'input changed during analysis')
    write(out/'MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
        for p in sorted(out.rglob('*')) if p.is_file()]})


def run(attempt,out,qualification=False):
    report,contract,out=begin(attempt,out)
    try:
        if qualification:
            _,fit,precision=numerical_rows(report,contract);groups={}
            for key in sorted({r['cell_id']+'/'+r['method'] for r in fit}):
                values=[r for r in fit if r['cell_id']+'/'+r['method']==key]
                groups[key]={'outputs':len(values),'fit_status_counts':dict(Counter(r['fit_status'] for r in values)),
                    'max_fit_mapping':max(r['fit_mapping'] for r in values),'max_independent_mapping':max(r['independent_mapping'] for r in values),
                    'max_gradient_SE':max(r['max_gradient_SE'] for r in values),'reference_diagnostic_passed':sum(r['reference_diagnostic_passed'] is True for r in values)}
            result={'schema':'local_shrinkage_numerical_qualification_v1','status':'completed','phase':report['phase'],
                'cases':len(contract['cases']),'outputs':len(report['outputs']),'risk_rows':len(precision),
                'all_rows_retained':True,'risk_qualified_count':sum(r['risk_qualified'] for r in precision),
                'max_N_times_KL_order_difference':max(r['N_times_KL_order_difference'] for r in precision),
                'max_PMF_order_difference':max(r['PMF_order_max_difference'] for r in precision),'fit_diagnostics':groups,
                'production_budget':report['budget'],'scientific_RNG_draws':0,'new_model_calls':0,
                'excluded':['joint_kl','scaled_joint_kl','risk_effects','direction','best_start_or_case'],
                'reference_mapping_is_diagnostic_only':True}
            csv_write(out/'FIT_DIAGNOSTICS.csv',fit);csv_write(out/'RISK_PRECISION.csv',precision)
        else:
            result,rows,fit,precision=validation_statistics(report,contract)
            result.update(schema='local_shrinkage_validation_analysis_v1',status='completed',phase='validation',scientific_RNG_draws=0,new_model_calls=0)
            csv_write(out/'CASE_METHOD_RISKS.csv',rows);csv_write(out/'FIT_DIAGNOSTICS.csv',fit);csv_write(out/'RISK_PRECISION.csv',precision)
            csv_write(out/'PRIMARY_CASE_VALUES.csv',[{'comparison':r['name'],'case_id':cid,'raw_difference':d,'scaled_difference':N*d}
                for r in result['primary_family'] for cid,d,N in zip(r['case_ids'],r['raw']['case_values'],[r['N']]*r['n'])])
            lines=['# Fixed-cell finite-N shrinkage risk','','Two original-start risks are averaged within each dataset. Four prespecified cell-specific comparisons.','',
                '|Comparison|Raw difference|Raw SE|Raw simultaneous interval|N × difference|N × SE|Scaled simultaneous interval|Direction|',
                '|---|---:|---:|---|---:|---:|---|---|']
            for r in result['primary_family']:
                a,b=r['raw'],r['scaled'];lines.append(f"|{r['name']}|{a['mean']:.9g}|{a['SE']:.6g}|[{a['CI'][0]:.9g}, {a['CI'][1]:.9g}]|{b['mean']:.9g}|{b['SE']:.6g}|[{b['CI'][0]:.9g}, {b['CI'][1]:.9g}]|{r['direction_supported']}|")
            lines += ['','All cases, starts and numerical failures remain. Reference stationarity is diagnostic, not an inclusion gate. Approximate dataset-level Student-t Bonferroni intervals are not teacher-population or equivalence claims.','',
                '|Cell|Configuration|Mean risk|Dataset SE|Mean N × risk|Scaled dataset SE|','|---|---|---:|---:|---:|---:|']
            for cid,data in result['descriptive'].items():
                for method,d in data['methods'].items():lines.append(f"|{cid}|{method}|{d['raw']['mean']:.9g}|{d['raw']['SE']:.6g}|{d['scaled']['mean']:.9g}|{d['scaled']['SE']:.6g}|")
            lines += ['','Group minus matched ridge is descriptive only (no extra equivalence test):']
            for cid,d in result['group_minus_ridge_descriptive'].items():lines.append(f"- {cid}: raw {d['raw']['mean']:.9g} (SE {d['raw']['SE']:.6g}); scaled {d['scaled']['mean']:.9g} (SE {d['scaled']['SE']:.6g}).")
            lines += ['','The nonzero fixed initialization and two-initialization baseline remain in the tables but are omitted from the main risk plot to preserve its local scale. N-dependent teachers and independent cells are not nested samples. Local-limit curves do not assert exact finite-N risk or global optimization.']
            with (out/'REPORT.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
            plot(out,result,contract)
        write(out/'SUMMARY.json',result);seal(out);return result
    except Exception:
        write(out/'FAILED.json',{'traceback':traceback.format_exc(),'scientific_RNG_draws':0,'new_model_calls':0});seal(out);raise


def plot(out,result,contract):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=contract['cells'];Ns=sorted({c['N'] for c in cells},reverse=True);maximum=max(c['S'] for c in cells)
    S=np.linspace(0,maximum*1.05,301);r=contract['theory_coefficients']['r_K8']
    curve={'S':S,'I':S/2,'integrated':np.full_like(S,1.5),'group_and_matched_ridge':.5*((1-r)**2*S+3*r*r)}
    markers={'integrated':'o','group8':'^','ridge':'s','I':'D'};colors={'integrated':'#476FA2','group8':'#117C83','ridge':'#B06D27','I':'#6E6D83'}
    fig,axes=plt.subplots(1,len(Ns),figsize=(6.2*len(Ns),4.9),squeeze=False,layout='constrained');points=[]
    for ax,N in zip(axes.flat,Ns):
        for key,label in [('I','I local limit'),('integrated','Integrated local limit'),('group_and_matched_ridge','Group / matched ridge local limit')]:ax.plot(S,curve[key],linestyle='--',lw=1.3,label=label)
        for cell in [c for c in cells if c['N']==N]:
            for method in (*METHODS,'I'):
                d=result['descriptive'][cell['cell_id']]['methods'][method]['scaled']
                ax.errorbar([cell['S']],[d['mean']],yerr=[d['SE']],fmt=markers[method],color=colors[method],
                    markerfacecolor=colors[method] if N==max(Ns) else 'none',capsize=3,label=method+' mean ±1SE')
                points.append({'cell_id':cell['cell_id'],'N':N,'S':cell['S'],'method':method,'scaled_mean':d['mean'],'scaled_SE':d['SE']})
        ax.set_title(f'N={N}: fixed-cell datasets');ax.set_xlabel('Local signal S');ax.set_ylabel('N × full-joint KL');ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys(),fontsize=7)
    fig.suptitle('First-order local limits and finite-N fixed-algorithm risks',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(out/('LOCAL_RISK_CURVES.'+ext),dpi=180)
    plt.close(fig)
    with (out/'PLOT_CURVES.npz').open('xb') as f:np.savez(f,**curve)
    csv_write(out/'PLOT_POINTS.csv',points)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--qualification',action='store_true');args=parser.parse_args()
    result=run(args.attempt,args.out,args.qualification);print(json.dumps({'schema':result['schema'],'status':result['status'],'scientific_RNG_draws':0,'new_model_calls':0}))
