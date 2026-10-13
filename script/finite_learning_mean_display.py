"""Analytic normal-limit display of central-event compensation, no new sampling."""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone
import numpy as np
from scipy.special import ndtr, xlogy
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DOC=ROOT/'docs/08_research_研究工作/有限样本学习与均值自由度_20261001'
OUT=ROOT/'artifacts/finite_learning/MEAN_DISPLAY'
OLD=ROOT/'artifacts/objective_strengthening/N'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def read(p):return json.loads(Path(p).read_text('utf-8'))
def write(p,x):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def verify(m):
    for n,h in m.items():assert sha(ROOT/n)==h,n
def freeze():
    assert not OUT.exists();OUT.mkdir(parents=True)
    paths=[Path(__file__),DOC/'PLAN_MEAN_DISPLAY.md',DOC/'MEAN_THEORY.md',DOC/'MEAN_REVIEW.md',OLD/'MOMENTS.json',OLD/'LIMITS.json']
    write(OUT/'FREEZE.json',dict(utc=datetime.now(timezone.utc).isoformat(),before_scalar_root_and_curve=True,
        files={rel(p):sha(p) for p in paths},new_samples=0,new_teacher_integrals=0))
def curves(beta,v):
    left=ndtr((-1-beta)/np.sqrt(v));right=ndtr((beta-1)/np.sqrt(v))
    return 1-left-right,left,right
def run():
    verify(read(OUT/'FREEZE.json')['files']);assert not (OUT/'RESULTS_FREEZE.json').exists()
    m=read(OLD/'MOMENTS.json');lim=next(r for r in read(OLD/'LIMITS.json')['limits'] if r['sign']==1)
    assert m['qualified'];alpha=m['gh'][-1]['values']['1.0']['alpha']
    vf,vt=lim['floor_variance'],lim['teacher_variance'];pt=lim['teacher_event'];rt=(1-pt)/2
    beta=brentq(lambda x:curves(x,vf)[0]-pt,0.,5.,xtol=1e-13,rtol=1e-14)
    pc,left,right=curves(beta,vf);qs=np.array([left,right]);p=np.full(2,rt)
    pair_brier=float(np.mean((qs-p)**2));pair_log=float(np.mean(xlogy(p,p/qs)+xlogy(1-p,(1-p)/(1-qs))))
    residual=abs(float(pc-pt));assert residual<1e-12 and pair_brier>0
    beta_grid=np.linspace(-1.2,1.2,1601);center,lo,hi=curves(beta_grid,vf)
    norm=float(np.max(abs(center+lo+hi-1)));assert norm<1e-14
    data=dict(input_source='old convergence-qualified moments; not certified interval constants',
        teacher_variance=vt,floor_variance=vf,alpha_floor=alpha,matched_beta=float(beta),
        matched_logit_mean_coefficient_t0=float(beta/alpha),teacher_center=pt,teacher_each_tail=rt,
        candidate_center=float(pc),candidate_left=float(left),candidate_right=float(right),
        center_probability_residual=residual,center_brier=float((pc-pt)**2),
        pair_brier=pair_brier,pair_logloss=pair_log,probability_sum_error=norm,
        new_random_samples=0,new_scientific_fits=0,new_teacher_integrals=0,
        deterministic_scalar_root_solves=1,analytic_grid_points=1601,
        trained_or_finite_dimension_model=False,beta=beta_grid.tolist(),center=center.tolist(),left=lo.tolist(),right=hi.tolist())
    write(OUT/'RESULTS.json',data)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10.7,4.1),layout='constrained')
    ax[0].plot(beta_grid,center,color='#226a9c',label='Floor-variance candidate')
    ax[0].axhline(pt,color='#c75032',ls='--',label='Teacher')
    ax[0].scatter([beta],[pc],color='#226a9c',zorder=4)
    ax[0].annotate(f'Matched central probability\nmean drift = {beta:.3f}',(beta,pc),xytext=(-1.05,.55),
        arrowprops={'arrowstyle':'->','color':'#444444'},fontsize=9)
    ax[0].set(title='Central event can be matched',ylabel='Probability of |S| <= 1',xlabel='Candidate mean drift of S')
    ax[0].legend(loc='upper center',fontsize=9)
    ax[1].plot(beta_grid,lo,color='#226a9c',label='Left tail S < -1')
    ax[1].plot(beta_grid,hi,color='#5b8f38',label='Right tail S > 1')
    ax[1].axhline(rt,color='#c75032',ls='--',label='Teacher: each tail')
    ax[1].axvline(beta,color='#777777',ls=':',lw=1)
    ax[1].scatter([beta,beta],[left,right],color=['#226a9c','#5b8f38'],zorder=4)
    ax[1].set(title='The same shift leaves both tails wrong',ylabel='Tail probability',xlabel='Candidate mean drift of S')
    ax[1].legend(loc='upper left',fontsize=9)
    fig.suptitle('Normal-limit illustration: free mean does not repair joint prediction',fontsize=12)
    for a in ax:a.grid(alpha=.15);a.set_xlim(-1.2,1.2)
    for suffix in ['png','svg']:fig.savefig(OUT/f'mean_compensation.{suffix}',dpi=180)
    plt.close(fig)
    paths=[OUT/'RESULTS.json',OUT/'mean_compensation.png',OUT/'mean_compensation.svg']
    write(OUT/'RESULTS_FREEZE.json',dict(source_freeze_sha256=sha(OUT/'FREEZE.json'),files={rel(p):sha(p) for p in paths}))
    print(json.dumps({k:v for k,v in data.items() if k not in ['beta','center','left','right']}))
def check():
    verify(read(OUT/'FREEZE.json')['files']);r=read(OUT/'RESULTS_FREEZE.json');verify(r['files'])
    assert r['source_freeze_sha256']==sha(OUT/'FREEZE.json')
    print(json.dumps(dict(passed=True,new_calculation=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run','check']);a=p.parse_args()
    {'freeze':freeze,'run':run,'check':check}[a.phase]()
