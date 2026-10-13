"""Full-label angular-risk ratios on an eight-row interval box; no file I/O."""
from fractions import Fraction as F
import time
import flint
from flint import arb,ctx
from r03_candidate_interval import aq,as_arb,record,endpoint_fraction,endpoint,label_bits

def up(x):return endpoint_fraction(endpoint(x.upper()))
def finite_risk(a_values,direction,*,step='1/8',N=88,bits=192,hard_seconds=120):
    assert flint.__version__=='0.8.0'
    old=ctx.prec;ctx.prec=bits;started=time.perf_counter()
    try:
        a=[as_arb(x) for x in a_values];u=[int(x) for x in direction['u_integer']]
        S=int(direction['S_integer']);dx=F(step)
        if len(a)!=8 or len(u)!=8 or sum(u)!=0 or sum(x*x for x in u)!=S or S<=0:raise ValueError('Fixed normalized direction')
        if any(not x>0 or not x<=1 for x in a) or dx<=0 or type(N)!=int or N<1:raise ValueError('Positive box and grid')
        coeff=[(i,j,aq(F((u[i]-u[j])**2,S))*a[i]*a[j]) for i in range(8) for j in range(i+1,8)]
        Q=[arb(0) for _ in range(256)];B=[arb(0) for _ in range(256)]
        mass=arb(0);completed=[]
        for k in range(-N,N+1):
            if time.perf_counter()-started>=hard_seconds:break
            z=aq(k*dx);w=aq(dx)*(-z*z/2).exp()/(2*arb.pi()).sqrt();mass+=w
            probs=[(1/(1+(ai*z).exp()),1/(1+(-ai*z).exp())) for ai in a]
            for y in range(256):
                bits_y=label_bits(y);like=arb(1)
                for i in range(8):like*=probs[i][bits_y[i]]
                residual=[arb(bits_y[i])-probs[i][1] for i in range(8)]
                pair=sum((c*residual[i]*residual[j] for i,j,c in coeff),arb(0))
                Q[y]+=w*like;B[y]+=w*like*pair
            completed.append(k)
        L=aq(N*dx);sec=1/(arb(1)/2).cos();alias_den=(2*arb.pi()/aq(dx)).exp()-1
        phi=(-L*L/2).exp()/(2*arb.pi()).sqrt();gauss=(arb(1)/2).exp()
        eQ=2*gauss*sec**8/alias_den+2*phi/L
        eN=2*gauss*sec**8*8*sec**2/alias_den+16*phi/L
        complete=len(completed)==2*N+1
        result=dict(schema='r03_risk_box_finite_v1',status='complete' if complete else 'budget_stop',
            configuration=dict(step_exact=str(dx),N=N,limit_exact=str(N*dx),precision_bits=bits,strip_exact='1'),
            a_enclosures=[record(x) for x in a],direction=dict(u_integer=u,S_integer=S),
            Q_finite=[record(x) for x in Q],N_finite=[record(x) for x in B],normal_mass=record(mass),
            analytic_errors=dict(Q=record(eQ),N=record(eN)),completed_indices=completed,
            counts=dict(Gaussian_nodes=len(completed),sigmoid_exponentials=16*len(completed),
                        label_finite_sums=512 if complete else 0,label_node_visits=256*len(completed),pair_terms=28*256*len(completed)),
            elapsed_seconds=time.perf_counter()-started,target_is_risk_ratio_box_not_training=True)
        if complete:result['composition']=compose(result)
        return result
    finally:ctx.prec=old

def compose(raw):
    if raw['status']!='complete':raise ValueError('Complete risk sum required')
    old=ctx.prec;ctx.prec=raw['configuration']['precision_bits']
    try:
        eQ=aq(up(as_arb(raw['analytic_errors']['Q'])));eN=aq(up(as_arb(raw['analytic_errors']['N'])))
        qs=[as_arb(x)+arb(0,eQ) for x in raw['Q_finite']]
        ns=[as_arb(x)+arb(0,eN) for x in raw['N_finite']]
        positive=all(q>0 for q in qs)
        return dict(schema='r03_risk_box_ratios_v1',a_enclosures=raw['a_enclosures'],direction=raw['direction'],
            Q_true_intervals=[record(x) for x in qs],N_true_intervals=[record(x) for x in ns],
            rho_true_intervals=[record(n/q) for n,q in zip(ns,qs)] if positive else [],
            all_Q_positive=positive,true_ratios_enclosed=positive,
            ratio_definition="N/Q = -Q_second_angle_derivative/Q; Q_first_angle_derivative=0",
            own_target_numerator_sum_zero=True,
            own_target_identity="sum_y A_y(z) P_y(z)=0 by independent Bernoulli centered cross-products; hence sum_y N_y=0 for every a",
            all_256_labels=True,whole_a_box_enclosed=True,new_grid_nodes=0,
            input_intervals_include_dependency_inflation=True)
    finally:ctx.prec=old

def self_test():
    # Exact finite Bernoulli pair identity; no Gaussian/project evaluation.
    ps=[F(i+1,10) for i in range(8)];u=list(range(-7,8,2));S=sum(x*x for x in u)
    total=F(0);mass=F(0)
    for y in range(256):
        yy=label_bits(y);p=F(1)
        for i in range(8):p*=ps[i] if yy[i] else 1-ps[i]
        pair=sum(F((u[i]-u[j])**2,S)*(yy[i]-ps[i])*(yy[j]-ps[j]) for i in range(8) for j in range(i+1,8))
        total+=p*pair;mass+=p
    assert mass==1 and total==0
    assert sum(F((u[i]-u[j])**2,S) for i in range(8) for j in range(i+1,8))==8
    return dict(passed=True,finite_label_cases=256,new_Gaussian_nodes=0,new_project_integrals=0)
