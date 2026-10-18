# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZHANG
"""Projected saved arithmetic: exact numerical functions, without runtime wrappers.

The numerical functions below are unchanged extracts of the frozen saved audit.
Only imports/constants and the public file wrapper are new. They reconstruct
full matrices from one-dimensional saved moments; no model, quadrature, RNG or
optimization is run. Preserved lambda-named data keys denote the common channel
called alpha in the accompanying notes.
"""
from collections import Counter
import numpy as np
PAIRS=((0,1),(0,2),(1,2))
DIRECTED=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))


class Audit:
    def __init__(self):self.failures=[];self.count=0;self.maximum_errors={};self.counts=Counter()
    def require(self,condition,label):
        self.count+=1
        if not bool(condition):self.failures.append(label)
        return bool(condition)
    def close(self,actual,expected,label,tolerance):
        actual=np.asarray(actual,dtype=float);expected=np.asarray(expected,dtype=float)
        # JSON's [] cannot distinguish a 0x0 matrix from a length-zero vector.
        if actual.shape==(0,) and expected.shape==(0,0):actual=actual.reshape(0,0)
        if not self.require(actual.shape==expected.shape,label+":shape"):return
        if not self.require(np.isfinite(actual).all() and np.isfinite(expected).all(),label+":finite"):return
        error=float(np.max(abs(actual-expected),initial=0.));self.maximum_errors[label]=error
        self.require(error<=tolerance,label+":absolute error "+str(error))
    def same(self,actual,expected,label):self.require(actual==expected,label)
    def section(self,label,fn):
        try:return fn()
        except Exception as error:self.require(False,label+": "+repr(error));return None


def one_dimensional_moments(statistics):
    """Retain raw mass; reject inconsistent/missing finite saved moments."""
    p=np.asarray(statistics["p"] if "p" in statistics else np.exp(statistics["log_normalizer"]),dtype=float)
    mu=np.asarray(statistics.get("mu",statistics.get("mean")),dtype=float)
    m2=np.asarray(statistics["second_moment"],dtype=float);var=np.asarray(statistics["variance"],dtype=float)
    z=np.asarray(statistics.get("z",statistics.get("log_normalizer_lambda")),dtype=float)
    if p.ndim!=1 or len(p)<2 or any(v.shape!=p.shape for v in (mu,m2,var,z)):raise ValueError("one-dimensional moment shape")
    if not all(np.isfinite(v).all() for v in (p,mu,m2,var,z)) or np.any(p<=0) or np.any(var<=0):raise ValueError("positive finite one-dimensional moments")
    first=np.asarray(statistics["mean_lambda"],dtype=float);second=np.asarray(statistics["log_normalizer_lambdalambda"],dtype=float)
    if any(x.shape!=p.shape or not np.isfinite(x).all() for x in (first,second)):raise ValueError("finite saved likelihood derivative fields")
    return dict(p=p,mu=mu,second_moment=m2,variance=var,z=z,mean_lambda=first,log_normalizer_lambdalambda=second)


def factorized_arrays(statistics,family,K):
    """One-dimensional tensor factors; no full observation-state enumeration."""
    if family not in ("F","L","A") or not isinstance(K,int) or K<1:raise ValueError("fixed family and positive finite integer K")
    d=one_dimensional_moments(statistics);p,mu,var,z=(d[k] for k in ("p","mu","variance","z"));S=len(p)
    mass=float(sum(p));m=float(p@mu);v=float(p@(mu*mu));w=float(p@var);A=float(p@d["second_moment"])
    zmean=float(p@z);z2=float(p@(z*z));muz=float(p@(mu*z))
    score_mean=np.r_[np.repeat(m*m*mass,3),3*zmean*mass*mass]
    score_second=np.full((4,4),0.)
    for i in range(3):
        score_second[i,i]=v*v*mass
        for j in range(i):score_second[i,j]=score_second[j,i]=v*m*m
        score_second[i,3]=score_second[3,i]=2*muz*m*mass+m*m*zmean
    score_second[3,3]=3*z2*mass*mass+6*zmean*zmean*mass
    J=v*v;information=3*z2
    P=0 if family=="F" else 3*S+(6 if family=="A" else 0);n=4+P
    G=np.zeros((P,P));B=np.zeros((3,P));R=np.eye(3)*(A*A-v*v)*mass
    for e in range(3):
        for f in range(e):R[e,f]=R[f,e]=w*m*m
    if P:
        for i in range(3):
            indices=np.arange(i*S,(i+1)*S);G[indices,indices]=p*var*mass*mass
            for e,(left,right) in enumerate(PAIRS):
                if i in (left,right):B[e,indices]=p*var*m*mass
        if family=="A":
            for j,(i,neighbor) in enumerate(DIRECTED):
                index=3*S+j;G[index,index]=w*v*mass
                for e,edge in enumerate(PAIRS):
                    if set(edge)=={i,neighbor}:B[e,index]=w*v*mass
                    elif i in edge:B[e,index]=w*m*m
                G[i*S:(i+1)*S,index]=G[index,i*S:(i+1)*S]=p*var*m*mass
                for k,(source,other) in enumerate(DIRECTED[:j]):
                    if source==i:G[index,3*S+k]=G[3*S+k,index]=w*m*m
    # Factor the actual saved negative-log-likelihood Hessian, rather than
    # inserting the Fisher identity being checked (especially its lambda block).
    likelihood=np.zeros((4,4));np.fill_diagonal(likelihood[:3,:3],2*A*mass*mass-mass**3-A*A*mass+v*v*mass)
    for e in range(3):
        for f in range(e):likelihood[e,f]=likelihood[f,e]=(mass-w)*m*m
    likelihood[:3,3]=likelihood[3,:3]=-2*float(p@d["mean_lambda"])*m*mass
    likelihood[3,3]=-3*float(p@d["log_normalizer_lambdalambda"])*mass*mass
    H=np.zeros((n,n));H[:4,:4]=likelihood;H[:3,:3]+=R/K
    if P:
        H[:3,4:]=-B/K;H[4:,:3]=-B.T/K;H[4:,4:]=G/K
    omega=np.zeros((n,n));omega[:4,:4]=score_second-np.outer(score_mean,score_mean)
    residual=np.zeros((n,n));residual[:3,:3]=R
    if P:residual[:3,4:]=-B;residual[4:,:3]=-B.T;residual[4:,4:]=G
    scalar=dict(v=v,w=w,q=A,J=J,I_lambda=information,K=K,r=J/(J+(1-J)/K if family!="A" else J+(1-v)**2/K),
        S=J+(1-J)/K if family!="A" else J+(1-v)**2/K)
    return dict(moments=d,scalar=scalar,H=H,likelihood_Hessian=likelihood,score_covariance=omega,B=B,G=G,residual_covariance=R,
        residual_full=residual,score_mean=score_mean,score_raw_second=score_second,score_cross=score_second[:3,3],
        input_mass=mass,input_mean=m,input_second=A,input_zmean=zmean,
        variance_consistency=float(np.max(abs(var-(d["second_moment"]-mu*mu)))))


def full_linear_risk(H,score_covariance,J,information):
    """Generic full solve, independent block solve and same-data plug-in map."""
    H=np.asarray(H,dtype=float);omega=np.asarray(score_covariance,dtype=float);n=len(H)
    if H.shape!=(n,n) or omega.shape!=H.shape or n<4:raise ValueError("complete Hessian/covariance dimensions")
    if not np.isfinite(H).all() or not np.isfinite(omega).all() or np.linalg.eigvalsh(H)[0]<=0:raise ValueError("finite positive definite Hessian")
    E=np.zeros((n,4));E[:4]=np.eye(4);full=np.linalg.solve(H,E)
    if n>4:
        nuisance=np.linalg.solve(H[4:,4:],H[4:,:4]);schur=H[:4,:4]-H[:4,4:]@nuisance
        prof=np.vstack([np.linalg.solve(schur,np.eye(4)),-nuisance@np.linalg.solve(schur,np.eye(4))])
    else:schur=H.copy();prof=np.linalg.solve(schur,np.eye(4))
    # Same N complete observations supply lambda's marginal score and theta's
    # score. This is not an independent second sample or a split-sample budget.
    keep=np.r_[0:3,4:n];oracle=np.zeros((n,4));oracle[keep]=np.linalg.solve(H[np.ix_(keep,keep)],E[keep])
    plugin=oracle.copy();plugin[3,3]=1/information
    plugin[keep]-=np.outer(np.linalg.solve(H[np.ix_(keep,keep)],H[keep,3]),plugin[3])
    covariance=full@omega[:4,:4]@full.T
    generation_covariance=covariance[:4,:4]
    covs=dict(joint=generation_covariance,oracle=(oracle@omega[:4,:4]@oracle.T)[:4,:4],plugin=(plugin@omega[:4,:4]@plugin.T)[:4,:4])
    operators=dict(joint=full,oracle=oracle,plugin=plugin)
    risk={};forcing=np.zeros((4,3));forcing[:3]=np.eye(3)*J;truth=np.vstack([np.eye(3),np.zeros((1,3))]);fisher=np.diag([J,J,J,information])
    for identity,cov in covs.items():
        shift=operators[identity][:4]@forcing-truth;bias=.5*shift.T@fisher@shift
        theta_variance=.5*J*np.diag(cov)[:3];channel_variance=.5*information*cov[3,3]
        risk[identity]=dict(bias_matrix=bias,bias_per_h_norm_squared=float(np.trace(bias)/3),theta_variance_per_coordinate=theta_variance,
            theta_variance_total=float(sum(theta_variance)),channel_variance=float(channel_variance),
            total_intercept=float(sum(theta_variance)+channel_variance))
    return dict(Schur=schur,full_operator=full,profile_operator=prof,full_covariance=covariance,generation_covariance=generation_covariance,
        oracle_generation_covariance=covs["oracle"],plugin_generation_covariance=covs["plugin"],plugin_operator=plugin,risk=risk)


def audit_case(row,statistics,K,a):
    prefix=row["rule"]+"/"+row["family"];f=factorized_arrays(statistics,row["family"],K);n=len(f["H"])
    a.same(row["dimension"],n,prefix+":full physical dimension");a.same(row["state_count"],len(f["moments"]["p"])**3,prefix+":deterministic state-node count")
    a.same(row["status"],"completed",prefix+":retained case status")
    fields=("log_normalizer","mean","second_moment","variance","log_normalizer_lambda","log_normalizer_lambdalambda","mean_lambda","mean_lambdalambda","second_moment_lambda")
    for key in fields:a.close(row["moments"][key],statistics[key],prefix+":frozen moment/"+key,0.)
    a.close(row["moments"]["p"],np.exp(statistics["log_normalizer"]),prefix+":raw input probability",0.)
    for name in ("input_mass","input_second"):a.close(f[name],1.,prefix+":"+name,1e-10)
    for name in ("input_mean","input_zmean","variance_consistency"):a.close(f[name],0.,prefix+":"+name,1e-10)
    a.close(f["score_mean"],np.zeros(4),prefix+":observed score mean",1e-10)
    a.close(f["score_cross"],np.zeros(3),prefix+":observed score cross",1e-10)
    info=np.diag([f["scalar"]["J"]]*3+[f["scalar"]["I_lambda"]])
    a.close(f["likelihood_Hessian"],info,prefix+":likelihood Fisher information identity",1e-10)
    a.close(row["state_sum"]["mass"],f["input_mass"]**3,prefix+":product raw mass",1e-10)
    for key in ("H","likelihood_Hessian","score_covariance","B","G","residual_covariance","score_mean","score_raw_second"):
        a.close(row["state_sum"][key],f[key],prefix+":state sum/factorized "+key,1e-10)
    for key in ("H","score_covariance","B","G","residual_covariance"):
        a.close(row["closed_form"][key],f[key],prefix+":saved closed/factorized "+key,1e-10)
    for key,value in f["scalar"].items():a.close(row["scalar"][key],value,prefix+":scalar/"+key,1e-10)
    linear=full_linear_risk(f["H"],f["score_covariance"],f["scalar"]["J"],f["scalar"]["I_lambda"])
    a.close(row["closed_form"]["Schur"],linear["Schur"][:3,:3],prefix+":complete Schur",1e-9)
    for key in ("full_operator","profile_operator","full_covariance","generation_covariance","oracle_generation_covariance","plugin_generation_covariance"):
        a.close(row["influence"][key],linear[key],prefix+":linear/"+key,1e-9)
    a.close(linear["profile_operator"],linear["full_operator"],prefix+":independent block versus full solve",1e-9)
    a.close(linear["plugin_operator"],linear["full_operator"],prefix+":same data plugin/joint first order",1e-9)
    local_mean=linear["full_operator"][:4,:3]*f["scalar"]["J"]
    a.close(row["influence"]["local_alternative_generation_mean_operator"],local_mean,prefix+":local mean",1e-9)
    for identity,q in linear["risk"].items():
        for key in ("bias_per_h_norm_squared","theta_variance_per_coordinate","theta_variance_total","channel_variance","total_intercept"):
            a.close(row["risk"][identity][key],q[key],prefix+":risk/"+identity+"/"+key,1e-9)
        a.close(q["bias_matrix"],np.eye(3)*q["bias_per_h_norm_squared"],prefix+":isotropic bias/"+identity,1e-9)
        a.close(q["theta_variance_per_coordinate"],np.repeat(.5*f["scalar"]["r"]**2,3),prefix+":all three variance coordinates/"+identity,1e-9)
        a.close(q["channel_variance"],0. if identity=="oracle" else .5,prefix+":common channel/"+identity,1e-9)
    a.close(linear["risk"]["joint"]["total_intercept"]-linear["risk"]["oracle"]["total_intercept"],.5,prefix+":oracle risk difference",1e-9)
    a.require(row["qualification"]["passed"] is True,prefix+":producer qualification retained")
    a.counts["factorized_cases"]+=1
    return dict(rule=row["rule"],family=row["family"],dimension=n,scalar=f["scalar"],risk=linear["risk"],
        minimum_H_eigenvalue=float(np.linalg.eigvalsh(f["H"])[0]),minimum_full_Schur_eigenvalue=float(np.linalg.eigvalsh(linear["Schur"])[0]))
