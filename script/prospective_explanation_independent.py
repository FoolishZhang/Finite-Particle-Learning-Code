"""Independent bounded numerical checks for the saved-object explanation.

The GL PMF evaluator is independent of the production GH PMF implementation.
Risk bookkeeping uses direct sums and both raw marginal cells; no PMF is
normalized. Scientific execution is restricted to a frozen six-GL/twelve-GH80
schedule. Manufactured tests never read the scientific inputs.
"""
from __future__ import annotations

import os
for _thread in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_thread]="1"

import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/prospective_explanation_v1_20261007"
OLD=ROOT/"artifacts/prospective_diagnostic_dev_v1_20261007"
GL_RULES=(("GL96_L10",96,10.,0.),("GL128_L10",128,10.,0.),("GL128_L10_rot_pi7",128,10.,math.pi/7))
MASS_TOL=2e-12
PMF_TOL=1e-12
KL_TOL=1e-10
IDENTITY_TOL=1e-12
GH80_TOL=1e-9


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def native(value):
    if isinstance(value,np.ndarray):return native(value.tolist())
    if isinstance(value,np.generic):return native(value.item())
    if isinstance(value,dict):return {str(k):native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return dict(nonfinite="positive_infinity" if value>0 else "negative_infinity" if value<0 else "nan")
    return value


def patterns(m):
    if not isinstance(m,int) or m<1 or m>20:raise ValueError("bounded positive pattern dimension required")
    return ((np.arange(1<<m)[:,None]>>np.arange(m-1,-1,-1))&1).astype(np.int8)


def _probability(p,m=None):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any(p<0) or p.sum()<=0:
        raise ValueError("a finite nonnegative positive-mass vector is required")
    if m is not None and len(p)!=(1<<m):raise ValueError("probability/pattern shape mismatch")
    return p


def gaussian_legendre_pmf(b,W,*,order,limit,rotation=0.,label_chunk=32):
    """Raw standard-Gaussian density-weighted GL tensor, axis 1 fastest.

    Rotation acts on nodes, not the model or weights; Gaussian density is
    invariant. No production model_pmf, GH rule, or mass renormalization is used.
    """
    b=np.asarray(b,dtype=float);W=np.asarray(W,dtype=float)
    if b.ndim!=1 or W.shape!=(len(b),2) or not np.isfinite(b).all() or not np.isfinite(W).all():
        raise ValueError("finite b[m], W[m,2] required")
    if not isinstance(order,int) or order<2 or not math.isfinite(limit) or limit<=0:
        raise ValueError("positive finite GL limit and integer order>=2 required")
    if not math.isfinite(rotation):raise ValueError("finite rotation required")
    x,w=np.polynomial.legendre.leggauss(order)
    x=limit*x;w=limit*w*np.exp(-x*x/2)/math.sqrt(2*math.pi)
    base=np.column_stack([np.repeat(x,order),np.tile(x,order)])
    c,s=math.cos(rotation),math.sin(rotation)
    R=np.array([[c,-s],[s,c]])
    nodes=base@R.T
    weights=(w[:,None]*w[None,:]).ravel()
    eta=b[None,:]+nodes@W.T
    yes=-np.logaddexp(0.,-eta);no=-np.logaddexp(0.,eta)
    Y=patterns(len(b));pmf=np.empty(len(Y))
    for start in range(0,len(Y),label_chunk):
        labels=Y[start:start+label_chunk]
        probabilities=np.exp(yes@labels.T+no@(1-labels).T)
        # Direct positive sum, not the production logsumexp path.
        for j in range(len(labels)):
            pmf[start+j]=math.fsum(float(v) for v in weights*probabilities[:,j])
    if not np.isfinite(pmf).all():raise FloatingPointError("nonfinite raw GL PMF")
    return dict(pmf=pmf,normal_nodes=nodes,normal_weights=weights,axis_nodes=x,axis_weights=w,
        normal_mass=math.fsum(float(v) for v in weights),pmf_mass=math.fsum(float(v) for v in pmf),
        strictly_positive=bool(np.all(pmf>0)),order=order,limit=float(limit),rotation=float(rotation),
        weights_renormalized=False,pmf_renormalized=False,
        work_counts=dict(PMF_calls=1,normal_nodes=order*order,labels=len(Y),
                         conditional_loglikelihood_values=order*order*len(Y),new_random_draws=0))


def raw_marginals(p,labels):
    """Each zero cell is independently summed, never computed as 1-p_one."""
    labels=np.asarray(labels);m=labels.shape[1];p=_probability(p,m)
    return np.array([[math.fsum(float(p[y]) for y in range(len(p)) if labels[y,j]==bit)
                      for bit in (0,1)] for j in range(m)])


def raw_kl(P,Q):
    P,Q=_probability(P),_probability(Q)
    if P.shape!=Q.shape:raise ValueError("matching KL shapes required")
    support=P>0
    if np.any(Q[support]==0):return math.inf
    return math.fsum(float(P[i])*(math.log(float(P[i]))-math.log(float(Q[i]))) for i in np.flatnonzero(support))


def risk_bookkeeping(P,Q,labels=None):
    """Exact finite-array identity on positive supports, not a causal split."""
    P,Q=_probability(P),_probability(Q)
    if P.shape!=Q.shape:raise ValueError("matching probability shapes required")
    m=len(P).bit_length()-1
    if len(P)!=(1<<m):raise ValueError("binary pattern vector required")
    Y=patterns(m) if labels is None else np.asarray(labels)
    if Y.shape!=(len(P),m):raise ValueError("invalid labels")
    p=raw_marginals(P,Y);q=raw_marginals(Q,Y);risk=raw_kl(P,Q)
    common=dict(P_mass=math.fsum(float(x) for x in P),Q_mass=math.fsum(float(x) for x in Q),
                P_marginals=p,Q_marginals=q,raw_KL=risk,probabilities_renormalized=False)
    if np.any(P<=0) or np.any(Q<=0) or np.any(p<=0) or np.any(q<=0) or not math.isfinite(risk):
        return dict(**common,decomposition_defined=False,
            undefined_reason="raw identity evaluated only for strictly positive finite probabilities; no inf-inf or zero division")
    logprodP=np.array([math.fsum(math.log(float(p[j,int(y[j])])) for j in range(m)) for y in Y])
    logprodQ=np.array([math.fsum(math.log(float(q[j,int(y[j])])) for j in range(m)) for y in Y])
    TC=math.fsum(float(P[i])*(math.log(float(P[i]))-float(logprodP[i])) for i in range(len(P)))
    perbit=np.array([math.fsum(float(p[j,k])*(math.log(float(p[j,k]))-math.log(float(q[j,k]))) for k in (0,1)) for j in range(m)])
    marginal=math.fsum(float(x) for x in perbit)
    J=math.fsum(float(P[i])*(math.log(float(Q[i]))-float(logprodQ[i])) for i in range(len(P)))
    residual=risk-(TC+marginal-J)
    return dict(**common,decomposition_defined=True,total_correlation=TC,marginal_KL_by_bit=perbit,
        marginal_KL_sum=marginal,dependence_score=J,identity_residual=residual,
        identity_qualified=abs(residual)<=IDENTITY_TOL,raw_terms_not_asserted_nonnegative=True,
        radial_angular_causal_decomposition=False)


def paired_bookkeeping(P,LOW,HIGH,labels=None):
    low=risk_bookkeeping(P,LOW,labels);high=risk_bookkeeping(P,HIGH,labels)
    if not low["decomposition_defined"] or not high["decomposition_defined"]:
        return dict(decomposition_defined=False,LOW=low,HIGH=high)
    P=_probability(P);LOW=_probability(LOW);HIGH=_probability(HIGH)
    direct=math.fsum(float(P[i])*(math.log(float(HIGH[i]))-math.log(float(LOW[i]))) for i in range(len(P)))
    marginal=low["marginal_KL_sum"]-high["marginal_KL_sum"]
    dependence=high["dependence_score"]-low["dependence_score"]
    return dict(decomposition_defined=True,paired_gain=direct,marginal_gain=marginal,
        dependence_score_gain=dependence,residual=direct-marginal-dependence,
        identity_qualified=abs(direct-marginal-dependence)<=IDENTITY_TOL,
        dependence_score_is_not_dependency_KL=True,radial_angular_causal_decomposition=False)


def independent_bernoulli_mle(labels):
    labels=np.asarray(labels)
    if labels.ndim!=2 or not len(labels) or not np.all((labels==0)|(labels==1)):
        raise ValueError("nonempty binary observations required")
    N,m=labels.shape
    p=np.array([math.fsum(float(x) for x in labels[:,j])/N for j in range(m)])
    Y=patterns(m)
    pmf=np.array([math.prod(float(p[j] if y[j] else 1-p[j]) for j in range(m)) for y in Y])
    train_nll=-math.fsum(math.log(math.prod(float(p[j] if y[j] else 1-p[j]) for j in range(m))) for y in labels)/N
    return dict(probability_one=p,pmf=pmf,N=N,boundary=bool(np.any((p==0)|(p==1))),
        empirical_NLL=train_nll,smoothed=False,probability_clipped=False,iterative_optimization=False,
        parameter_class="independent Bernoulli distributions; no finite bias-box claim at boundary")


def recode_probability(P,signs):
    signs=np.asarray(signs);m=len(signs);P=_probability(P,m)
    if not np.all((signs==1)|(signs==-1)):raise ValueError("fixed +/-1 signs required")
    mask=sum(1<<(m-1-j) for j in range(m) if signs[j]<0)
    permutation=np.arange(1<<m,dtype=int)^mask
    result=np.empty_like(P);result[permutation]=P
    return result,permutation


def reweight_saved_labels(teacher_pmfs,signs,label_Hangle,label_gradient,direction):
    """Direct teacher-weight sums on saved labels, no population-mode search."""
    H=np.asarray(label_Hangle,dtype=float);g=np.asarray(label_gradient,dtype=float)
    if H.shape[0]==1 and H.ndim==4:H=H[0]
    if g.shape[0]==1 and g.ndim==3:g=g[0]
    direction=np.asarray(direction,dtype=float);m=len(direction)
    if H.shape!=(1<<m,m,m) or g.shape!=(1<<m,2*m):raise ValueError("saved label derivative shapes disagree")
    matrices=[];gradients=[];curvatures=[]
    for P in np.asarray(teacher_pmfs):
        qr,_=recode_probability(P,signs)
        matrix=np.empty((m,m));gradient=np.empty(2*m)
        for i in range(m):
            for j in range(m):matrix[i,j]=math.fsum(float(qr[y])*float(H[y,i,j]) for y in range(len(qr)))
        for j in range(2*m):gradient[j]=math.fsum(float(qr[y])*float(g[y,j]) for y in range(len(qr)))
        curvature=math.fsum(float(direction[i])*float(matrix[i,j])*float(direction[j]) for i in range(m) for j in range(m))
        matrices.append(matrix);gradients.append(gradient);curvatures.append(curvature)
    return dict(population_Hangle=np.asarray(matrices),population_gradient=np.asarray(gradients),
                population_direction_curvature=np.asarray(curvatures),fixed_direction=direction.copy(),direction_reselected=False)


def numerical_sign(value,tau):
    return "positive" if value>tau else "negative" if value< -tau else "unresolved"


def assess_gh80(H,gradient,d,expected_H,expected_gradient,main_tau,expected_sign=None):
    m=len(d)
    curvature=math.fsum(float(d[i])*float(H[i,j])*float(d[j]) for i in range(m) for j in range(m))
    primary=math.fsum(float(d[i])*float(expected_H[i,j])*float(d[j]) for i in range(m) for j in range(m))
    primary_sign=numerical_sign(primary,main_tau) if expected_sign is None else expected_sign
    if primary_sign not in ("positive","negative","unresolved"):raise ValueError("invalid primary readout sign")
    independent_sign=numerical_sign(curvature,main_tau);same=independent_sign==primary_sign
    sign_ok=primary_sign=="unresolved" or same
    he=float(np.max(abs(H-expected_H)));ge=float(np.max(abs(gradient-expected_gradient)))
    return dict(direction_curvature=curvature,expected_direction_curvature=primary,
        matrix_max_difference=he,gradient_max_difference=ge,Hangle_qualified=he<=GH80_TOL,gradient_qualified=ge<=GH80_TOL,
        primary_sign=primary_sign,independent_sign=independent_sign,directional_sign_equal=same,
        directional_sign_qualified=sign_ok,primary_unresolved_is_not_upgraded=primary_sign=="unresolved",
        qualified=bool(he<=GH80_TOL and ge<=GH80_TOL and sign_ok))


def gh80_teacher_reference(theta_recoded,signs,teacher_main_pmf,direction,expected_H,expected_gradient,main_tau,expected_sign=None):
    """One old-kernel K=infinity GH80 call; never a finite-Laplace replay."""
    import rankone_stationary_kernel as kernel
    theta=np.asarray(theta_recoded,dtype=float);m=len(signs)
    if theta.shape!=(2*m,) or not np.isfinite(theta).all():raise ValueError("fixed finite recoded theta required")
    qr,permutation=recode_probability(teacher_main_pmf,signs)
    d=np.asarray(direction,dtype=float)
    if d.shape!=(m,) or not np.isfinite(d).all():raise ValueError("fixed saved direction required")
    raw=kernel.rankone_stationary(theta[:m],theta[m:],{"teacher_GH60":qr},K_values=("inf",),
                                  normal_rule="hermite",normal_order=80,compute_hessian=True)
    H=raw["Hangle"][0,0];gradient=raw["gradient"][0,0]
    return dict(Hangle=H,gradient=gradient,direction=d.copy(),
        **assess_gh80(H,gradient,d,expected_H,expected_gradient,main_tau,expected_sign),
        theta_recoded=theta.copy(),teacher_weights_recoded=qr,old_to_new_pattern_index=permutation,
        normal_nodes=raw["normal_nodes"],normal_weights=raw["normal_weights"],
        work_counts=raw["work_counts"],fresh_rankone_GH80_calls=1,fresh_finite_Laplace_calls=0,
        original_diagnostic_direction_not_training_kick=True,population_negative_mode_not_selected=True,
        teacher_probability_is_original_saved_GH60=True,strict_error_certificate=False)


class Checks:
    def __init__(self):self.hashes={};self.failures=[];self.maxima={};self.count=0
    def require(self,condition,context):
        self.count+=1
        if not bool(condition):self.failures.append(context)
    def same(self,actual,expected,context):self.require(native(actual)==native(expected),context)
    def close(self,actual,expected,field,context,tol=IDENTITY_TOL):
        a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
        equal_inf=(a.shape==b.shape and np.array_equal(a,b))
        error=0. if equal_inf else float(np.max(abs(a-b))) if a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() else None
        self.require(error is not None and error<=tol,context+":"+field)
        if error is not None and error>self.maxima.get(field,{}).get("error",-1.):self.maxima[field]=dict(error=error,context=context)
        return error
    def bind(self,path,expected=None):
        path=Path(path);digest=sha(path);name=rel(path)
        if name in self.hashes:self.same(digest,self.hashes[name],name+":unchanged")
        self.hashes[name]=digest
        if expected is not None:
            if isinstance(expected,dict):expected=expected['sha256']
            self.same(digest,expected,name+":hash")
        return digest


class SavedTree:
    """Lazy selector for old/new dev.persist trees; no training-tape replay."""
    def __init__(self,stem,checks,expected=None):
        self.metadata=Path(stem).with_suffix('.json');self.record=read(self.metadata)
        checks.bind(self.metadata,None if expected is None else expected.get(rel(self.metadata)))
        raw=ROOT/self.record['raw']['path'];checks.bind(raw,self.record['raw'])
        if expected is not None:checks.bind(raw,expected[rel(raw)])
        self.archive=np.load(raw,allow_pickle=False)
    def get(self,*keys):
        value=self.record['tree']
        for key in keys:value=value[key]
        def decode(x):
            if isinstance(x,dict) and '__ndarray__' in x:
                a=self.archive[x['__ndarray__']].copy()
                if list(a.shape)!=x['shape'] or str(a.dtype)!=x['dtype']:raise ValueError('stored array metadata differs')
                return a
            if isinstance(x,dict) and '__nonfinite_float__' in x:return float(x['__nonfinite_float__'])
            if isinstance(x,dict):return {k:decode(v) for k,v in x.items()}
            if isinstance(x,list):return [decode(v) for v in x]
            return x
        return decode(value)
    def close(self):self.archive.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def compare_unit_arithmetic(unit,teacher,checks,expected_inputs):
    uid=unit['id'];folder=OLD/'units'/uid
    with SavedTree(folder/'inputs',checks,expected_inputs) as inp, SavedTree(folder/'anchor',checks,expected_inputs) as anc, \
         SavedTree(folder/'diagnosis',checks,expected_inputs) as diag, SavedTree(folder/'risk',checks,expected_inputs) as risk, \
         SavedTree(OUT/'units'/uid/'explanation',checks) as produced:
        q,labels=inp.get('q'),inp.get('labels')
        signs,theta=anc.get('signs'),anc.get('theta_recoded')
        d=diag.get('numerics','direction')
        checks.same(produced.get('fixed_empirical_direction'),d,uid+':unchanged_diagnostic_direction')
        checks.require(np.asarray(produced.get('fixed_empirical_direction')).tobytes()==np.asarray(d).tobytes(),uid+':direction_bytes')
        checks.same(produced.get('theta_recoded'),theta,uid+':unchanged_anchor')
        checks.same(produced.get('signs'),signs,uid+':unchanged_signs')
        for output,key in (('original_A','full_raise'),('original_B','ref_raise'),('original_full_level','full_level'),
                           ('original_full_reasons','full_reasons'),('original_ref_reasons','ref_reasons'),
                           ('original_diagnostic_thresholds','thresholds')):
            checks.same(produced.get(output),diag.get(key),uid+':unchanged_'+output)
        checks.same(produced.get('empirical_frequency'),q,uid+':unchanged_original_frequency')
        checks.same(produced.get('recoded_teacher_pmfs'),np.array([recode_probability(P,signs)[0] for P in teacher]),uid+':unchanged_teacher_weights')
        H=[];g=[];c=[]
        for rule in ('C','D'):
            values=reweight_saved_labels(teacher,signs,diag.get('reference_only','raw',rule,'label_Hangle'),
                                         diag.get('reference_only','raw',rule,'label_gradient'),d)
            H.append(values['population_Hangle']);g.append(values['population_gradient']);c.append(values['population_direction_curvature'])
        H,g,c=np.asarray(H),np.asarray(g),np.asarray(c)
        for field,value in dict(population_Hangle=H,population_gradient=g,population_direction_curvature=c).items():
            checks.close(produced.get(field),value,'label_reweight.'+field,uid)
        readout=produced.get('population_curvature_readout')
        main=float(c[1,1]);disagreement=float(np.max(abs(c-main)));tau=max(1e-8,10*disagreement)
        sign='positive' if np.all(c>tau) else 'negative' if np.all(c< -tau) else 'unresolved'
        for field,value in dict(main_value=main,max_difference=disagreement,tau_num=tau).items():
            checks.close(readout[field],value,'curvature_readout.'+field,uid)
        checks.same(readout['sign'],sign,uid+':all_eight_variant_sign')
        empirical={}
        for key,phase,ki in (('8','finite_extra',0),('64','finite_extra',1),('inf','reference_only',0)):
            empirical[key]=np.asarray([diag.get(phase,'raw',rule,'gradient')[0,ki] for rule in ('C','D')])
            checks.same(produced.get('empirical_gradients',key),empirical[key],uid+':unchanged_empirical_gradient_'+key)
            for ri,rule in enumerate(('C','D')):
                vector=empirical[key][ri];m=len(vector)//2
                for block,part in (('total',vector),('bias',vector[:m]),('row_amplitude',vector[m:])):
                    norms=produced.get('empirical_gradient_norms',key,rule,block)
                    checks.close(norms['L2'],math.hypot(*(float(x) for x in part)),'gradient_norm.L2',uid+'/'+key+'/'+rule+'/'+block)
                    checks.close(norms['Linf'],float(np.max(abs(part))),'gradient_norm.Linf',uid+'/'+key+'/'+rule+'/'+block)
        channel=produced.get('gradient_channel');norm=math.hypot(*(float(x) for x in empirical['64'][1]))
        checks.close(channel['empirical_F64_D_L2'],norm,'gradient_channel.norm',uid)
        checks.same(channel['defined'],norm>1e-10,uid+':gradient_channel_defined')
        if norm>1e-10:
            direction=-empirical['64'][1]/norm
            checks.close(channel['direction_recoded'],direction,'gradient_channel.direction',uid)
            derivatives=np.array([[math.fsum(float(x)*float(y) for x,y in zip(g[ci,vi],direction)) for vi in range(4)] for ci in range(2)])
            checks.close(channel['directional_derivatives'],derivatives,'gradient_channel.derivatives',uid)
            signed=np.r_[signs,signs]
            checks.close(channel['direction_signed_original'],signed*direction,'gradient_channel.signed_direction',uid)
            lower=np.r_[np.full(len(signs),-2.),np.full(len(signs),.05)];upper=np.full(len(theta),2.)
            ratios=[(upper[j]-theta[j])/direction[j] if direction[j]>0 else (lower[j]-theta[j])/direction[j]
                    for j in range(len(theta)) if direction[j]!=0]
            checks.close(channel['maximum_forward_box_step'],min(ratios),'gradient_channel.box_step',uid)
            channel_main=float(derivatives[1,1]);channel_diff=float(np.max(abs(derivatives-channel_main)));channel_tau=max(1e-8,10*channel_diff)
            channel_sign='positive' if np.all(derivatives>channel_tau) else 'negative' if np.all(derivatives< -channel_tau) else 'unresolved'
            for field,v in dict(main_value=channel_main,max_difference=channel_diff,tau_num=channel_tau).items():
                checks.close(channel['readout'][field],v,'gradient_channel.readout_'+field,uid)
            checks.same(channel['readout']['sign'],channel_sign,uid+':gradient_channel_all_variants_sign')
        endpoints=risk.get('pmfs');saved_gain=risk.get('paired_gain');books={}
        for arm in ('LOW','HIGH','REF'):
            books[arm]=[]
            output=produced.get('risk_decomposition',arm)
            for vi in range(4):
                b=risk_bookkeeping(teacher[vi],endpoints[arm][vi]);books[arm].append(b)
                mapping=dict(raw_KL='raw_KL',teacher_TC='total_correlation',marginal_KL_sum='marginal_KL_sum',marginal_KL_by_bit='marginal_KL_by_bit',
                    dependence_score='dependence_score',teacher_marginals='P_marginals',model_marginals='Q_marginals',identity_residual='identity_residual')
                for field,own in mapping.items():checks.close(output[vi][field],b[own],'risk_account.'+field,uid+'/'+arm+f'/{vi}')
                checks.require(b['identity_qualified'],uid+'/'+arm+f'/{vi}:raw_identity')
        gains=[];output=produced.get('gain_decomposition')
        for vi in range(4):
            value=paired_bookkeeping(teacher[vi],endpoints['LOW'][vi],endpoints['HIGH'][vi]);gains.append(value)
            for field,own in dict(direct_gain='paired_gain',marginal_gain='marginal_gain',dependence_score_gain='dependence_score_gain',identity_residual='residual').items():
                checks.close(output[vi][field],value[own],'gain_account.'+field,uid+f'/{vi}')
            checks.close(output[vi]['saved_gain'],saved_gain[vi],'gain_account.saved_gain',uid+f'/{vi}')
            checks.close(value['paired_gain'],saved_gain[vi],'gain_account.old_risk_identity',uid+f'/{vi}')
        mle=independent_bernoulli_mle(labels);ind=produced.get('independent_baseline')
        checks.close(ind['one_probabilities'],mle['probability_one'],'IND.MLE_probabilities',uid,0.)
        checks.close(ind['pmf'],mle['pmf'],'IND.MLE_product',uid)
        checks.close(ind['empirical_NLL'],mle['empirical_NLL'],'IND.empirical_NLL',uid)
        checks.same(ind['boundary'],mle['boundary'],uid+':IND_boundary')
        checks.same(ind['no_smoothing'],True,uid+':IND_no_smoothing')
        checks.same(ind['analytic_fit'],True,uid+':IND_closed_form')
        mle_risk=np.array([raw_kl(P,mle['pmf']) for P in teacher])
        checks.close(ind['raw_KL'],mle_risk,'IND.raw_risk',uid)
        checks.require(not mle['smoothed'],uid+':IND_no_smoothing')
        return dict(unit=unit,theta=theta,signs=signs,direction=d,H=H,g=g,curvatures=c,readout=readout,
                    raw_identity_max_error=max(abs(b['identity_residual']) for rows in books.values() for b in rows),
                    paired_identity_max_error=max(abs(v['residual']) for v in gains),
                    independent_MLE=mle,independent_MLE_risk=mle_risk)


def teacher_distance_checks(gl_pmfs,old_pmfs):
    """All GL/GH variants are retained; main GL rule is fixed128, unrotated."""
    gl=np.asarray(gl_pmfs);gh=np.asarray(old_pmfs)
    if gl.shape!=(2,3,256) or gh.shape!=(2,4,256):raise ValueError('fixed D1/D2 GL/GH shapes required')
    primary=gl[:,1]
    gl_KL=np.array([raw_kl(gl[1,i],gl[0,i]) for i in range(3)])
    gh_KL=np.array([raw_kl(gh[1,i],gh[0,i]) for i in range(4)])
    main=float(gl_KL[1]);values=np.r_[gl_KL,gh_KL]
    differences=dict(GL_mass_error=float(np.max(abs(gl.sum(axis=2)-1))),
        GL_PMF_max_difference=float(np.max(abs(gl-primary[:,None,:]))),
        saved_GH_PMF_max_difference_from_GL128=float(np.max(abs(gh-primary[:,None,:]))),
        GL_KL_max_difference=float(np.max(abs(gl_KL-main))),
        all_GL_and_saved_GH_KL_max_difference=float(np.max(abs(values-main))))
    gates=dict(raw_probabilities_positive=bool(np.all(gl>0)),raw_probabilities_finite=bool(np.isfinite(gl).all()),
        GL_mass=differences['GL_mass_error']<=MASS_TOL,GL_PMF=differences['GL_PMF_max_difference']<=PMF_TOL,
        GL_vs_saved_GH_PMF=differences['saved_GH_PMF_max_difference_from_GL128']<=PMF_TOL,
        KL_all_variants=differences['all_GL_and_saved_GH_KL_max_difference']<=KL_TOL)
    pinsker={str(N):(math.sqrt(N*main/2) if main>=0 else None) for N in (1024,4096)}
    return dict(primary_rule='GL128_L10',direction='D(P_D2 || P_D1)',main_numerical_KL=main,
        GL_KL_variants=gl_KL,saved_GH_KL_variants=gh_KL,GL_raw_PMF_masses=gl.sum(axis=2),saved_GH_raw_PMF_masses=gh.sum(axis=2),
        differences=differences,gates=gates,
        qualified=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v],
        Pinsker_numerical_substitutions=pinsker,strict_numerical_upper_bound=False,
        one_factor_witness='D1 is a permissible rank-one distribution; numeric KL is an approximate witness, not a certified upper bound',
        classic_product_KL_and_Pinsker_are_not_new_theory=True,reviewer_sandbox_script_obtained_or_run=False)


def verify_freeze(checks):
    path=OUT/'FREEZE.json';checks.bind(path);frozen=read(path)
    checks.same(frozen['schema'],'prospective_explanation_freeze_v1','freeze_schema')
    expected=dict(GL_orders=[96,128,128],GL_limit=10.,GL_rotations=['0','0','pi/7'],GL_PMF_calls=6,
        GH_anchor_order=80,GH_K_infinity_calls=12,normal_mass_gate=2e-12,PMF_gate=1e-12,KL_gate=1e-10,
        anchor_matrix_gradient_gate=1e-9,automatic_order_escalation=False)
    checks.same(frozen['independent_config'],expected,'fixed_independent_budget_and_gates')
    required=[Path(__file__),Path(__file__).with_name('prospective_explanation_independent_test.py'),
              ROOT/'script/rankone_stationary_kernel.py',ROOT/'docs/08_research_研究工作/ProspectiveExplanation_v1_20261007/CONTRACT.md']
    checks.require(all(rel(p) in frozen['sources'] for p in required),'required_source_bindings')
    for name,binding in frozen['sources'].items():checks.bind(ROOT/name,binding)
    config=read(OLD/'CONFIG.json');checks.same(config['units'],frozen['config']['original_units'],'twelve_frozen_units')
    expected_inputs={}
    oldfreeze=sha(OLD/'FREEZE.json')
    for name in ('INPUTS_LOCK','DIAGNOSIS_LOCK','ENDPOINTS_LOCK','EVALUATION_LOCK'):
        lock=read(OLD/(name+'.json'));checks.same(lock['freeze_sha256'],oldfreeze,'old_'+name+'_freeze')
        expected_inputs.update(lock['files'])
    producer=read(OUT/'RESULTS.json');checks.bind(OUT/'RESULTS.json')
    checks.same(producer['freeze_sha256'],sha(path),'producer_freeze')
    checks.same(producer['source_sha256'],sha(ROOT/'script/prospective_explanation_compute.py'),'producer_source')
    checks.same(producer['status'],'completed_saved_object_derivation','producer_complete')
    checks.same([r['unit'] for r in producer['unit_artifacts']],[u['id'] for u in config['units']],'all_producer_units')
    for row in producer['unit_artifacts']:
        for name in ('json','npz'):checks.bind(ROOT/row[name]['path'],row[name])
    if checks.failures:raise ValueError('Source/freeze verification failed: '+repr(checks.failures))
    return frozen,config,expected_inputs


def perform(*,reintegrate,arrays,progress):
    tick=time.perf_counter();checks=Checks();frozen,config,expected_inputs=verify_freeze(checks)
    teachers=[];rules=None
    for i in range(3):
        with SavedTree(OLD/f'teacher_D{i+1}',checks,expected_inputs) as tree:
            pmfs=tree.get('pmfs');this_rules=tree.get('rules')
            checks.same(this_rules,frozen['config']['teacher_rule_order'],'teacher_rule_order')
            teachers.append(pmfs);rules=this_rules
    checks.same(config['teachers'][0]['b'],config['teachers'][1]['b'],'D1_D2_same_bias')
    checks.same(config['teachers'][0]['a'],config['teachers'][1]['a'],'D1_D2_same_row_norms')
    checks.require(config['teachers'][0]['phi']!=config['teachers'][1]['phi'],'D1_D2_different_angles')
    if reintegrate:
        gl=np.empty((2,3,256));masses=np.empty((2,3))
        for ti in range(2):
            teacher=config['teachers'][ti];b=np.asarray(teacher['b']);a=np.asarray(teacher['a']);phi=np.asarray(teacher['phi'])
            W=a[:,None]*np.column_stack([np.cos(phi),np.sin(phi)])
            for ri,(name,order,limit,rotation) in enumerate(GL_RULES):
                progress['active']=dict(teacher=ti+1,rule=name)
                progress['GL_PMF_calls_started']+=1
                value=gaussian_legendre_pmf(b,W,order=order,limit=limit,rotation=rotation)
                progress['GL_PMF_calls_completed']+=1
                gl[ti,ri]=value['pmf'];masses[ti,ri]=value['normal_mass']
                prefix=f'GL_D{ti+1}_{ri}'
                arrays[prefix+'_pmf']=value['pmf'];arrays[prefix+'_nodes']=value['normal_nodes'];arrays[prefix+'_weights']=value['normal_weights']
        arrays['gl_pmfs']=gl;arrays['gl_normal_mass']=masses
    else:
        gl=arrays['gl_pmfs'];masses=arrays['gl_normal_mass']
    for ti in range(2):
        for ri,(_,order,_,_) in enumerate(GL_RULES):
            prefix=f'GL_D{ti+1}_{ri}'
            checks.same(arrays[prefix+'_pmf'],gl[ti,ri],'saved_GL_PMF_copy')
            checks.same(arrays[prefix+'_nodes'].shape,(order*order,2),'saved_GL_node_shape')
            checks.close(masses[ti,ri],math.fsum(float(x) for x in arrays[prefix+'_weights']),'GL.normal_mass','GL',0.)
    distance=teacher_distance_checks(gl,np.asarray(teachers[:2]))
    distance['normal_rule_mass_error']=float(np.max(abs(masses-1)))
    distance['gates']['normal_rule_mass']=distance['normal_rule_mass_error']<=MASS_TOL
    distance['qualified']=all(distance['gates'].values())
    distance['failed_gates']=[k for k,v in distance['gates'].items() if not v]
    arithmetic=[];anchor_checks=[]
    for unit in config['units']:
        value=compare_unit_arithmetic(unit,teachers[unit['teacher']],checks,expected_inputs)
        uid=unit['id'];prefix='GH80_'+uid
        expected_H=value['H'][1,1];expected_g=value['g'][1,1]
        if reintegrate:
            progress['active']=dict(unit=uid,rule='GH80_K_infinity')
            progress['GH_K_infinity_calls_started']+=1
            fresh=gh80_teacher_reference(value['theta'],value['signs'],teachers[unit['teacher']][1],value['direction'],
                                         expected_H,expected_g,value['readout']['tau_num'],value['readout']['sign'])
            progress['GH_K_infinity_calls_completed']+=1
            for key in ('Hangle','gradient','direction','theta_recoded','teacher_weights_recoded','normal_nodes','normal_weights'):
                arrays[prefix+'_'+key]=fresh[key]
            arrays[prefix+'_finite_K_greater_than_one']=np.asarray(fresh['work_counts']['finite_K_greater_than_one'])
        checks.same(arrays[prefix+'_finite_K_greater_than_one'],0,uid+':no_finite_Laplace')
        checks.same(arrays[prefix+'_theta_recoded'],value['theta'],uid+':GH80_fixed_theta')
        checks.same(arrays[prefix+'_direction'],value['direction'],uid+':GH80_fixed_d')
        checks.same(arrays[prefix+'_teacher_weights_recoded'],recode_probability(teachers[unit['teacher']][1],value['signs'])[0],uid+':GH80_original_GH60_teacher')
        assessed=assess_gh80(arrays[prefix+'_Hangle'],arrays[prefix+'_gradient'],value['direction'],expected_H,expected_g,
                            value['readout']['tau_num'],value['readout']['sign'])
        anchor_checks.append(dict(unit=uid,**assessed,main_tau=value['readout']['tau_num'],
            kernel_rule='existing frozen rankone kernel, K=infinity, Hermite80',new_finite_Laplace_calls=0,
            original_diagnostic_direction_not_training_kick=True,population_mode_not_selected=True))
        arithmetic.append(dict(unit=uid,raw_identity_max_error=value['raw_identity_max_error'],
            paired_identity_max_error=value['paired_identity_max_error'],independent_MLE_boundary=value['independent_MLE']['boundary'],
            independent_MLE_probability_one=value['independent_MLE']['probability_one'],independent_MLE_risk=value['independent_MLE_risk']))
        print(json.dumps(dict(checked_unit=uid,GH80_numerically_qualified=assessed['qualified'],new_integration=reintegrate)),flush=True)
    if reintegrate:
        arrays['executed_GL_PMF_calls']=np.asarray(progress['GL_PMF_calls_completed'])
        arrays['executed_GH_K_infinity_calls']=np.asarray(progress['GH_K_infinity_calls_completed'])
    checks.same(arrays['executed_GL_PMF_calls'],6,'fixed_six_GL_calls')
    checks.same(arrays['executed_GH_K_infinity_calls'],12,'fixed_twelve_GH80_calls')
    for name,digest in list(checks.hashes.items()):checks.bind(ROOT/name,digest)
    all_qualified=distance['qualified'] and all(r['qualified'] for r in anchor_checks)
    return dict(passed=not checks.failures,status='completed' if all_qualified else 'completed_with_numerical_qualification_limits',
        source_sha256=sha(__file__),test_sha256=sha(Path(__file__).with_name('prospective_explanation_independent_test.py')),
        freeze_sha256=sha(OUT/'FREEZE.json'),producer_results_sha256=sha(OUT/'RESULTS.json'),input_hashes=checks.hashes,
        failures=checks.failures,checks=checks.count,numerical_maxima=checks.maxima,teacher_distance=distance,
        anchor_GH80_checks=anchor_checks,arithmetic_checks=arithmetic,all_numerical_comparisons_qualified=all_qualified,
        quadrature_counts=dict(GL_PMF_calls=6,GH_K_infinity_calls=12,finite_Laplace_calls=0,
                               interpretation='calls in initial frozen run; --check reuses saved arrays and makes zero new calls'),
        scope=dict(saved_objects_post_hoc=True,new_independent_validation=False,units=12,
            joint_risk_bookkeeping_cells=144,paired_risk_decompositions=48,closed_form_MLE_rechecks=12,
            teacher_weight_Hangle_reweightings=96,teacher_weight_gradient_reweightings=96,
            new_data_units=0,new_random_draws=0,new_optimizer_updates=0,new_diagnostic_rules=0,
            original_A_B_and_directions_unchanged=True,original_d_is_not_the_training_kick=True,
            only_fixed_direction_not_all_population_modes=True,no_radial_angle_or_trajectory_causal_attribution=True,
            GL_PMF_implementation_independent_of_production_GH=True,GH80_uses_previously_frozen_rankone_formula=True,
            rigorous_quadrature_or_Pinsker_numeric_certificate=False,reviewer_sandbox_obtained_or_executed=False,
            prior_and_training_tapes_not_decoded=True),
        seconds=time.perf_counter()-tick,environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test',action='store_true');mode.add_argument('--run',action='store_true')
    mode.add_argument('--check',action='store_true',help='check stored results and algebra only; no GL/GH integration')
    args=parser.parse_args()
    if args.self_test:
        import prospective_explanation_independent_test as tests
        result=tests.run_tests();print(json.dumps(native(result),indent=2,allow_nan=False));return 0 if result['passed'] else 1
    target=OUT/'INDEPENDENT_RESULTS.json';rawpath=OUT/'INDEPENDENT_ARRAYS.npz'
    if args.run and any(p.exists() for p in (target,rawpath,OUT/'INDEPENDENT_FAILURE.json',OUT/'INDEPENDENT_PARTIAL_ARRAYS.npz')):
        raise FileExistsError('independent output or failed attempt exists; no overwrite, implicit retry or reintegration')
    previous=None;arrays={}
    if args.check:
        previous=read(target)
        if previous['source_sha256']!=sha(__file__):raise ValueError('independent source changed after receipt')
        if sha(rawpath)!=previous['arrays_artifact']['sha256']:raise ValueError('saved independent arrays changed')
        with np.load(rawpath,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    progress=dict(GL_PMF_calls_started=0,GL_PMF_calls_completed=0,GH_K_infinity_calls_started=0,GH_K_infinity_calls_completed=0)
    try:
        result=perform(reintegrate=args.run,arrays=arrays,progress=progress)
        if args.run:
            with rawpath.open('xb') as f:np.savez_compressed(f,**arrays)
            result['arrays_artifact']=dict(path=rel(rawpath),sha256=sha(rawpath),bytes=rawpath.stat().st_size)
            result['created_utc']=datetime.now(timezone.utc).isoformat()
            result['initial_run_actual_counts']=progress
            with target.open('x',encoding='utf-8',newline='\n') as f:
                json.dump(native(result),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        elif result['passed']:
            for key in ('passed','status','source_sha256','test_sha256','freeze_sha256','producer_results_sha256','input_hashes',
                        'failures','checks','numerical_maxima','teacher_distance','anchor_GH80_checks','arithmetic_checks',
                        'all_numerical_comparisons_qualified','quadrature_counts','scope'):
                if native(result[key])!=previous[key]:raise ValueError('stored independent check differs: '+key)
        print(json.dumps(dict(passed=result['passed'],all_numerical_qualified=result['all_numerical_comparisons_qualified'],
            new_GL_calls=progress['GL_PMF_calls_completed'],new_GH80_calls=progress['GH_K_infinity_calls_completed'],seconds=result['seconds'])),flush=True)
        return 0 if result['passed'] else 1
    except BaseException as error:
        if args.run and not (OUT/'INDEPENDENT_FAILURE.json').exists():
            partial=OUT/'INDEPENDENT_PARTIAL_ARRAYS.npz'
            if arrays and not partial.exists():
                with partial.open('xb') as f:np.savez_compressed(f,**arrays)
            with (OUT/'INDEPENDENT_FAILURE.json').open('x',encoding='utf-8',newline='\n') as f:
                json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(),error=str(error),traceback=traceback.format_exc(),
                    actual_counts=progress,source_sha256=sha(__file__),automatic_retry=False),f,indent=2,ensure_ascii=False);f.write('\n')
        raise


if __name__=='__main__':raise SystemExit(main())
