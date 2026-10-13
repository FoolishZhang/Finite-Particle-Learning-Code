"""Post-hoc conditional-KL decomposition from sealed PMFs, with no new samples."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/structure_tradeoff_v2';C=BASE/'C';R=BASE/'R2';OUT=BASE/'analysis'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def main():
    assert not (BASE/'MANIFEST.json').exists(),'Final delivery is sealed; use the read-only delivery check'
    assert read(C/'STATE.json')['status']=='complete'
    for n,h in read(C/'RESULTS_FREEZE.json')['files'].items():assert sha(ROOT/n)==h
    OUT.mkdir(exist_ok=True);rows=[];source={};maxerror=0.;minimum_a=math.inf;minimum_i=math.inf
    d=16;m=8;a,b=np.meshgrid(np.arange(9),np.arange(9),indexing='ij');a=a.ravel();b=b.ravel();n=a+b
    lg=gammaln(m+1)-gammaln(a+1)-gammaln(m-a+1)+gammaln(m+1)-gammaln(b+1)-gammaln(m-b+1)
    lt=gammaln(d+1)-gammaln(n+1)-gammaln(d-n+1);ind=np.eye(17)[n]
    for profile in ['null','mild','contrast']:
        refpath=C/'reference'/f'{profile}_d16.json';p=np.array(read(refpath)['mean']);pn=p@ind
        D=float(p@np.log(p/(pn[n]*np.exp(lg-lt))));results=[]
        source[refpath.relative_to(ROOT).as_posix()]=sha(refpath)
        for stream in range(8):
            path=C/'local'/f'{profile}_s{stream}.npz';source[path.relative_to(ROOT).as_posix()]=sha(path)
            with np.load(path) as z:qg=z['group_count_pmf'];qn=z['total_count_pmf'];old=z['losses']
            assert qg.shape==(32,2,81) and qn.shape==(32,2,17)
            I=np.sum(pn*np.log(pn/qn),axis=-1)
            A=np.sum(p*np.log((p/pn[n])/(qg/qn[:,:,n])),axis=-1)
            reconstructed=np.stack([I+A,D+I,D-A],axis=-1)
            maxerror=max(maxerror,float(np.max(abs(reconstructed-old))))
            minimum_a=min(minimum_a,float(A.min()));minimum_i=min(minimum_i,float(I.min()))
            results.append(np.stack([I,A,D-A],axis=-1))
        allrows=np.concatenate(results)
        for ki,k in enumerate([1000,100000]):
            vals=allrows[:,ki].mean(0)
            rows.append(dict(profile=profile,d=d,K=k,model_KL_reference=D,count_KL_mean=float(vals[0]),
                allocation_KL_mean=float(vals[1]),single_minus_group_mean=float(vals[2]),
                identity='single_minus_group = model_KL - allocation_KL',posthoc_derived_not_new_endpoint=True))
    assert maxerror<1e-10 and minimum_a>=-1e-10 and minimum_i>=-1e-10
    with (OUT/'ALLOCATION.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # The score curve uses the explicitly post-hoc 90-reading simultaneous sensitivity.
    sensitivity=table(C/'JOINT_SENSITIVITY_90.csv')
    print('joint sensitivity fields',list(sensitivity[0]),flush=True)
    paired=[r for r in sensitivity if r['reading']=='single_minus_group' and int(r.get('d',16))==16]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,profile in zip(axes,['null','mild','contrast']):
        current=sorted([r for r in paired if r['profile']==profile],key=lambda r:int(r['K']))
        assert len(current)==5,(profile,len(current))
        x=np.array([int(r['K']) for r in current]);y=np.array([float(r['mean']) for r in current])
        lo=np.array([float(r['lower']) for r in current]);hi=np.array([float(r['upper']) for r in current])
        ax.plot(x,y,color='#315d82',lw=1)
        for xi,yi,l,h,r in zip(x,y,lo,hi,current):
            qualified=r.get('precision_pass','True')=='True'
            ax.errorbar(xi,yi,yerr=[[yi-l],[h-yi]],fmt='o',mfc='#315d82' if qualified else 'white',mec='#315d82',ecolor='#315d82',capsize=3)
        D=0. if profile=='null' else next(r['model_KL_reference'] for r in rows if r['profile']==profile)
        ax.axhline(0,color='0.5',lw=.8);ax.axhline(D,color='#bd613b',ls='--',label='Exact-risk limit (reference)')
        ax.set(xscale='log',yscale='symlog',xlabel='Particles K',title=profile)
        ax.set_yscale('symlog',linthresh=1e-4);ax.set_ylim(-3.2,.006);ax.grid(alpha=.16)
    axes[0].set_ylabel('Single - group full-pattern score (nats)')
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('At nonzero heterogeneity, finite-score ranking reverses with particle budget (d = 16)',fontsize=12)
    fig.text(.5,-.055,'Post-hoc 90-reading intervals. Open markers: original precision gate failed. Lines connect tested budgets; no minimum crossover K is estimated.',ha='center',fontsize=9)
    fig.savefig(BASE/'figures/score_budget.png',dpi=220,bbox_inches='tight');fig.savefig(BASE/'figures/score_budget.svg',bbox_inches='tight');plt.close(fig)
    psvg=BASE/'figures/score_budget.svg';psvg.write_text('\n'.join(s.rstrip() for s in psvg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    source[(C/'JOINT_SENSITIVITY_90.csv').relative_to(ROOT).as_posix()]=sha(C/'JOINT_SENSITIVITY_90.csv')
    result=dict(passed=True,source_files=source,posthoc_algebraic_diagnostic=True,new_samples=0,new_fits=0,
        bank_count=768,metric_rows=6,conditional_chain_rule_max_error=maxerror,
        minimum_bank_allocation_KL=minimum_a,minimum_bank_count_KL=minimum_i,
        positivity_tolerance=1e-10,new_independent_success_claim=False,
        plot_intervals='post-hoc R2/C 90-reading sensitivity; not a new independent comparison',
        reference_uncertainty_not_in_t_intervals=True)
    write(OUT/'ALLOCATION_AUDIT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files'}))
if __name__=='__main__':main()
