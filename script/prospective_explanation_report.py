"""Descriptive author readout and figure from locked explanation summaries.

No training, quadrature, reclassification of old alarms or causal attribution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/prospective_explanation_v1_20261007'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def derive():
    s=read(OUT/'SUMMARY.json')['derived']; independent=read(OUT/'INDEPENDENT_RESULTS.json')
    assert independent['passed'] and independent['all_numerical_comparisons_qualified']
    rows=s['rows'];assert len(rows)==12
    empirical=[r for r in rows if r['empirical_curvature'][1]<-1e-8]
    alarms=[r for r in rows if r['A']]
    adverse=[r for r in rows if r['saved_delta_R']<0]
    positive=[r for r in rows if r['saved_delta_R']>0]
    mean=lambda key:float(np.mean([r['main_gain_account'][key] for r in rows]))
    norms=lambda K:[r['empirical_gradient_norms'][K]['D']['total']['L2'] for r in rows]
    g64=norms('64');gi=norms('inf');g8=norms('8')
    witness=independent['teacher_distance']
    facts=dict(units=12,empirical_negative_directions=len(empirical),
        empirical_negative_but_population_positive=[r['unit'] for r in empirical if r['population_curvature']['sign']=='positive'],
        empirical_negative_and_population_negative=[r['unit'] for r in empirical if r['population_curvature']['sign']=='negative'],
        FULL_alarm_population_positive=[r['unit'] for r in alarms if r['population_curvature']['sign']=='positive'],
        FULL_alarm_population_negative=[r['unit'] for r in alarms if r['population_curvature']['sign']=='negative'],
        population_negative_but_HIGH_worse_than_LOW=[r['unit'] for r in adverse if r['population_curvature']['sign']=='negative'],
        population_direction_counts={key:sum(r['population_curvature']['sign']==key for r in rows) for key in ('positive','negative','unresolved')},
        population_risk_derivative_along_fixed_empirical_HIGH_bias_amplitude_descent_counts={key:sum(r['gradient_channel']['sign']==key for r in rows) for key in ('positive','negative','unresolved')},
        gradient_norm_L2_ranges=dict(K8=[min(g8),max(g8)],K64=[min(g64),max(g64)],integrated=[min(gi),max(gi)]),
        radial_bias_directional_derivative_range=[min(r['gradient_channel']['main_value'] for r in rows),max(r['gradient_channel']['main_value'] for r in rows)],
        mean_total_gain=mean('direct_gain'),mean_marginal_gain=mean('marginal_gain'),mean_dependence_score_gain=mean('dependence_score_gain'),
        HIGH_worse_than_LOW_units=[r['unit'] for r in adverse],
        HIGH_worse_despite_improved_marginals=[r['unit'] for r in adverse if r['main_gain_account']['marginal_gain']>0 and r['main_gain_account']['dependence_score_gain']<0],
        HIGH_better_with_worse_marginals=[r['unit'] for r in positive if r['main_gain_account']['marginal_gain']<0],
        main_independent_Bernoulli_risk_range=[min(r['independent_MLE_risk'][1] for r in rows),max(r['independent_MLE_risk'][1] for r in rows)],
        mean_independent_Bernoulli_risk=float(np.mean([r['independent_MLE_risk'][1] for r in rows])),
        joint_endpoints_better_than_same_data_independent_baseline=sum(r['main_risk_accounts'][arm]['raw_KL']<r['independent_MLE_risk'][1] for r in rows for arm in ('LOW','HIGH','REF')),
        independent_GL_D2_to_D1_KL=witness['main_numerical_KL'],Pinsker_numerical_substitutions=witness['Pinsker_numerical_substitutions'],
        maximum_risk_identity_residual=max(r['maximum_finite_risk_identity_residual'] for r in rows),
        new_observations=0,new_random_priors=0,new_optimizer_updates=0,causal_attribution=False,
        numerical_upper_bound_certified=False,original_d_not_training_kick=True,
        source_SUMMARY_sha256=sha(OUT/'SUMMARY.json'),source_INDEPENDENT_RESULTS_sha256=sha(OUT/'INDEPENDENT_RESULTS.json'))
    assert abs(facts['mean_total_gain']-facts['mean_marginal_gain']-facts['mean_dependence_score_gain'])<1e-12
    return facts,rows

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'prospective-explanation-v1'})
    fig,ax=plt.subplots(1,2,figsize=(12.2,4.6),layout='constrained')
    x=np.array([r['empirical_curvature'][1] for r in rows])*1e3
    y=np.array([r['population_curvature']['main_value'] for r in rows])*1e3
    for idx,r in enumerate(rows):
        ax[0].scatter(x[idx],y[idx],marker='^' if r['A'] else 'o',s=60,
            color='#B34232' if r['saved_delta_R']<0 else '#267D9D',edgecolor='white',linewidth=.7)
        ax[0].annotate(str(idx+1),(x[idx],y[idx]),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax[0].axhline(0,color='#777777',linewidth=.8);ax[0].axvline(0,color='#777777',linewidth=.8)
    ax[0].set(xlabel='Empirical fixed-direction curvature (x 1e-3)',ylabel='Population fixed-direction curvature (x 1e-3)',title='Same saved direction; no population-mode selection')
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([],[],marker='^',linestyle='',color='#666666',label='FULL alarm'),
        Line2D([],[],marker='o',linestyle='',color='#666666',label='No FULL alarm'),
        Line2D([],[],marker='o',linestyle='',color='#B34232',label='HIGH worse than LOW'),
        Line2D([],[],marker='o',linestyle='',color='#267D9D',label='HIGH better than LOW')],fontsize=8,loc='lower right')
    order=np.arange(1,13);margin=np.array([r['main_gain_account']['marginal_gain'] for r in rows])*1e3
    dep=np.array([r['main_gain_account']['dependence_score_gain'] for r in rows])*1e3
    gain=np.array([r['saved_delta_R'] for r in rows])*1e3
    ax[1].bar(order-.18,margin,.34,color='#CB963C',label='Marginal-error gain')
    ax[1].bar(order+.18,dep,.34,color='#438DAC',label='Dependence-score gain')
    ax[1].scatter(order,gain,marker='D',s=20,color='#222222',label='Total gain',zorder=3)
    ax[1].axhline(0,color='#777777',linewidth=.8);ax[1].set_xticks(order)
    ax[1].set(xlabel='Original unit number (full IDs in table)',ylabel='Risk-gain accounting (x 1e-3 nat)',title='Algebraic risk accounting, not coordinate causation')
    ax[1].legend(fontsize=8,loc='upper left')
    for a in ax:a.grid(alpha=.18);a.set_axisbelow(True)
    fig.suptitle('Post-hoc explanation of 12 locked development units',fontsize=12)
    for suffix in ('png','svg'):
        path=OUT/('explanation_readout_final.'+suffix)
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180)
    plt.close(fig)

def main(check):
    facts,rows=derive();path=OUT/'AUTHOR_FINDINGS_FINAL.json'
    if check:
        saved=read(path);assert saved['facts']==facts and saved['source_sha256']==sha(__file__)
        for name,digest in saved['figures'].items():assert sha(ROOT/name)==digest
    else:
        if path.exists():raise FileExistsError(path)
        plot(rows);save(path,dict(source_sha256=sha(__file__),facts=facts,
            figures={(OUT/('explanation_readout_final.'+ext)).relative_to(ROOT).as_posix():sha(OUT/('explanation_readout_final.'+ext)) for ext in ('png','svg')}))
    print(json.dumps(dict(passed=True,check=check,new_quadrature_calls=0,new_training=0,units=len(rows))))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run',action='store_true');g.add_argument('--check',action='store_true');a=p.parse_args();main(a.check)
