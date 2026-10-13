"""Render the fixed angular evidence, including all qualification failures."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/joint_dependence_v1_20261007'
r=json.loads((OUT/'ANGULAR_RESULTS.json').read_text('utf-8'))
x=np.array([z['epsilon'] for z in r['rows']]);ok=np.array([z['qualified'] for z in r['rows']])
y=[-np.array([z['K2_difference'] for z in r['rows']]),np.array([z['evaluations'][1]['risk'] for z in r['rows']]),np.array([z['derivative_over_epsilon4'] for z in r['rows']])]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
for ax,vals in zip(axes,y):
 ax.plot(x,vals,color='#94a3b8',lw=1,zorder=1)
 ax.scatter(x[ok],vals[ok],s=45,color='#146b78',label='Passes fixed numerical gate',zorder=3)
 ax.scatter(x[~ok],vals[~ok],s=60,facecolors='white',edgecolors='#b56524',marker='s',label='Gate not passed',zorder=3)
 ax.set_xlabel('Loading scale');ax.set_xticks(x);ax.grid(axis='y',alpha=.2)
axes[0].set_title('(a) Finite K=2 objective decrease');axes[0].set_ylabel('F2(truth) minus F2(compressed)');axes[0].set_yscale('log');axes[0].set_ylim(y[0].min()/2,y[0].max()*2)
axes[1].set_title('(b) Exact-model observation KL');axes[1].set_yscale('log');axes[1].set_ylabel('nats / eight-bit observation')
axes[2].set_title('(c) Directional derivative / scale^4');axes[2].axhline(r['geometry']['K2_derivative_limit'],color='#334155',ls='--',lw=1,label='Derived small-scale limit')
axes[2].legend(fontsize=8,loc='best')
fig.suptitle('Fixed one-dimensional marginals; 5% angular compression',fontsize=13)
paths=[]
for ext in ['png','svg']:
 q=OUT/('angular_evidence.'+ext);fig.savefig(q);paths.append(q)
plt.close(fig)
(OUT/'FIGURE.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results_sha256':hashlib.sha256((OUT/'ANGULAR_RESULTS.json').read_bytes()).hexdigest(),'outputs':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'new_scientific_computation':False,'failed_qualifications_displayed':True},indent=2)+'\n',encoding='utf-8')
