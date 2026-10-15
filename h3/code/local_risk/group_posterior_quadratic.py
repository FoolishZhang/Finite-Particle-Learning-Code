"""Fixed coefficient centered-origin-Hessian control for group-posterior banks.

No RNG, optimizer, teacher information or file writes. This changes a fixed SAA
scalar while preserving its ideal expectation with exact actual-proposal moments.
It is not an independent test of the origin Hessian that it cancels by design.
"""
from __future__ import annotations
import time
import numpy as np
from group_posterior_objective import PreparedGroupBanks as BasePreparedGroupBanks, PAIRS

SCHEMA = "group_posterior_quadratic_v1"
CONTROL = "uniform_logratio_plus_centered_origin_hessian"
EXPONENTS = np.asarray(((1,1,0),(1,0,1),(0,1,1)),dtype=int)


def statistics(values, gradients):
    R=len(values)
    return {"value":float(values.mean()),"gradient":gradients.mean(axis=0),
        "replicate_values":values,"replicate_gradients":gradients,
        "replicate_se":float(values.std(ddof=1)/np.sqrt(R)) if R>1 else None,
        "replicate_gradient_se":gradients.std(axis=0,ddof=1)/np.sqrt(R) if R>1 else None}


class PreparedGroupBanks(BasePreparedGroupBanks):
    """Same constructor and borrowed/frozen-array contract as the base object."""
    def __post_init__(self):
        super().__post_init__()
        if self._constant_local is None:
            raise ValueError("origin-Hessian group-posterior extension requires constant local log-normalizers [3,S]")
        self._origin_hessian_cache=None
        self._origin_diagnostics_cache={}

    def origin_hessian_control(self, *, pattern_chunk=64):
        """D=Sbar*Sbar'-mean(s*s')+(1-1/K)Cov_q(s|actual O).

        Each q product uses exponents 0/1/2 and its actual mean/second moment.
        Pattern weights may be empirical frequencies or offline population PMF.
        No known Fisher/H_K/teacher coefficients are used in this calculation.
        """
        if isinstance(pattern_chunk,bool) or not isinstance(pattern_chunk,(int,np.integer)) or pattern_chunk<1:
            raise ValueError("positive integer pattern_chunk required")
        if self.proposal_mean is None:raise ValueError("actual proposal moments required for centered Hessian")
        if self._origin_hessian_cache is not None:
            D,cov=self._origin_hessian_cache
            return {"D":D,"conditional_score_covariance":cov,"cache_hit":True,
                    "particle_pattern_terms_this_call":0,"moment_pattern_terms_this_call":0,"seconds_this_call":0.}
        started=time.perf_counter();D=np.zeros((self.R,self.B,3,3));cov=np.zeros((3,3))
        work=0
        # At K1 both empirical terms are identical and the covariance factor is 0.
        for start in range(0,len(self.weights),int(pattern_chunk)):
            stop=min(start+int(pattern_chunk),len(self.weights))
            states=self.state_indices[start:stop];weight=self.weights[start:stop]
            if self.K>1:
                x=[np.moveaxis(self.x_group[...,g,states[:,g]],2,3) for g in range(3)]
                score=[x[i]*x[j] for i,j in PAIRS]
                average=[s.mean(axis=-1) for s in score]
            mu=np.stack([self.proposal_mean[g,states[:,g]] for g in range(3)],axis=-1)
            nu=np.stack([self.proposal_second_moment[g,states[:,g]] for g in range(3)],axis=-1)
            mean_score=[mu[:,i]*mu[:,j] for i,j in PAIRS]
            for a in range(3):
                for b in range(a,3):
                    if self.K>1:
                        empirical=average[a]*average[b]-(score[a]*score[b]).mean(axis=-1)
                        contribution=np.einsum('rbp,p->rb',empirical,weight)
                        D[...,a,b]+=contribution
                        if a!=b:D[...,b,a]+=contribution
                    product=np.ones(len(states))
                    for g,power in enumerate(EXPONENTS[a]+EXPONENTS[b]):
                        if power==1:product*=mu[:,g]
                        elif power==2:product*=nu[:,g]
                    expected=float(weight@(product-mean_score[a]*mean_score[b]))
                    cov[a,b]+=expected
                    if a!=b:cov[b,a]+=expected
            if self.K>1:work+=self.R*self.B*self.K*len(states)
        D+=(1.-1./self.K)*cov
        if not np.isfinite(D).all() or not np.isfinite(cov).all():
            raise FloatingPointError("nonfinite centered origin Hessian; no clipping")
        D.setflags(write=False);cov.setflags(write=False)
        self._origin_hessian_cache=(D,cov)
        return {"D":D,"conditional_score_covariance":cov,"cache_hit":False,
                "particle_pattern_terms_this_call":work,"moment_pattern_terms_this_call":len(self.weights),
                "seconds_this_call":time.perf_counter()-started}

    def origin_diagnostics(self,include_triangle=True,*,pattern_chunk=64):
        """Uncentered raw derivatives on fixed banks; no Hessian control applied.

        Only constant-local-weight group posteriors have uniform particle weights
        at I. This independent bank diagnostic never inserts H_K/T/b values.
        """
        if not isinstance(include_triangle,(bool,np.bool_)):raise ValueError("include_triangle must be boolean")
        if isinstance(pattern_chunk,bool) or not isinstance(pattern_chunk,(int,np.integer)) or pattern_chunk<1:
            raise ValueError("positive integer pattern_chunk required")
        key=bool(include_triangle)
        if key in self._origin_diagnostics_cache:
            return {**self._origin_diagnostics_cache[key],"origin_diagnostic_cache_hit":True,
                    "origin_diagnostic_particle_pattern_terms":0,"origin_diagnostic_seconds_this_call":0.}
        started=time.perf_counter();H=np.zeros((self.R,self.B,3,3));triangle=np.zeros((self.R,self.B))
        for start in range(0,len(self.weights),int(pattern_chunk)):
            stop=min(start+int(pattern_chunk),len(self.weights));states=self.state_indices[start:stop];weight=self.weights[start:stop]
            x=[np.moveaxis(self.x_group[...,g,states[:,g]],2,3) for g in range(3)]
            score=[x[i]*x[j] for i,j in PAIRS];average=[s.mean(axis=-1) for s in score]
            for a,(i,j) in enumerate(PAIRS):
                for b in range(a,3):
                    if a==b:t=1.-x[i]**2-x[j]**2
                    else:
                        other=sorted((set(PAIRS[a])|set(PAIRS[b]))-(set(PAIRS[a])&set(PAIRS[b])))
                        t=-x[other[0]]*x[other[1]]
                    local=-t.mean(axis=-1)-(score[a]*score[b]).mean(axis=-1)+average[a]*average[b]
                    contribution=np.einsum('rbp,p->rb',local,weight)
                    H[...,a,b]+=contribution
                    if a!=b:H[...,b,a]+=contribution
            if key:
                r123=(x[0]*x[0]-1)*(x[1]*x[1]-1)*(x[2]*x[2]-1)
                local=-r123.mean(axis=-1)-2*average[0]*average[1]*average[2]
                for a,(i,j) in enumerate(PAIRS):
                    shared=3-i-j
                    mixed=(x[shared]*x[shared]-1)*score[a]
                    local+=average[a]*mixed.mean(axis=-1)
                triangle+=np.einsum('rbp,p->rb',local,weight)
        if not np.isfinite(H).all() or (key and not np.isfinite(triangle).all()):
            raise FloatingPointError("nonfinite raw origin diagnostic")
        rep=H.mean(axis=1);H.setflags(write=False)
        result={"raw_Hessian_bank":H,"raw_Hessian_replicate":rep,"raw_Hessian_mean":rep.mean(axis=0),
            "raw_Hessian_replicate_SE":rep.std(axis=0,ddof=1)/np.sqrt(self.R) if self.R>1 else None,
            "include_triangle":key,"origin_diagnostic_cache_hit":False,
            "origin_diagnostic_particle_pattern_terms":self.R*self.B*self.K*len(self.weights),
            "origin_diagnostic_seconds_this_call":time.perf_counter()-started,
            "uncertainty_unit":"complete_independent_replicate_including_all_rejection_trial_nets",
            "scope":"uncentered_raw_origin_derivatives; no_theory_values_inserted; not_centered_control_confirmation"}
        if key:
            rep_triangle=triangle.mean(axis=1);triangle.setflags(write=False)
            result.update(raw_mixed_triangle_bank=triangle,raw_mixed_triangle_replicate=rep_triangle,
                raw_mixed_triangle_mean=float(rep_triangle.mean()),
                raw_mixed_triangle_replicate_SE=float(rep_triangle.std(ddof=1)/np.sqrt(self.R)) if self.R>1 else None)
        for value in result.values():
            if isinstance(value,np.ndarray):value.setflags(write=False)
        self._origin_diagnostics_cache[key]=result
        return result

    def evaluate(self,C,*,control="uniform_logratio",pattern_chunk=64,return_pattern_details=False):
        if control=="raw":
            return super().evaluate(C,control=control,pattern_chunk=pattern_chunk,return_pattern_details=return_pattern_details)
        if control not in ("uniform_logratio",CONTROL):raise ValueError("raw or centered uniform_logratio control required")
        if return_pattern_details:
            raise ValueError("centered-origin extension does not provide per-pattern details; raw details remain supported")
        base=super().evaluate(C,control="uniform_logratio",pattern_chunk=pattern_chunk,return_pattern_details=False)
        kernel=self.origin_hessian_control(pattern_chunk=pattern_chunk)
        C=np.asarray(C,float);theta=C[[0,0,1],[1,2,2]];D=kernel['D']
        correction=.5*np.einsum('i,rbij,j->rb',theta,D,theta)
        correction_gradient=np.einsum('rbij,j->rbi',D,theta)
        bank=base['bank_values']-correction;gradient=base['bank_gradients']-correction_gradient
        if not np.isfinite(bank).all() or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite quadratic controlled scalar/gradient")
        result=dict(base)
        for name in ('value','gradient','bank_values','bank_gradients','replicate_values','replicate_gradients','replicate_se','replicate_gradient_se'):
            result['base_uniform_logratio_'+name]=base[name]
        result.update(statistics(bank.mean(axis=1),gradient.mean(axis=1)))
        result.update(schema=SCHEMA,control=CONTROL,bank_values=bank,bank_gradients=gradient,
            origin_hessian_D=D,center_conditional_score_covariance=kernel['conditional_score_covariance'],
            quadratic_correction_bank_values=correction,quadratic_correction_bank_gradients=correction_gradient,
            quadratic_cache_hit=kernel['cache_hit'],
            quadratic_cache_particle_pattern_terms_this_call=kernel['particle_pattern_terms_this_call'],
            control_construction_particle_pattern_terms=kernel['particle_pattern_terms_this_call'],
            control_construction_moment_pattern_terms=kernel['moment_pattern_terms_this_call'],
            quadratic_cache_seconds_this_call=kernel['seconds_this_call'],
            quadratic_cache_work_scope="one_cache_particle_pattern_kernel; six_symmetric_pair_contractions; not_measured_CPU_operations",
            origin_hessian_qualification="conditional_zero_mean_with_exact_actual_proposal_moments; numerical_centering_error_changes_value_and_gradient; not_independent_origin_Hessian_confirmation")
        return result


QuadraticPreparedGroupBanks=PreparedGroupBanks
QuadraticGroupBanks=PreparedGroupBanks
