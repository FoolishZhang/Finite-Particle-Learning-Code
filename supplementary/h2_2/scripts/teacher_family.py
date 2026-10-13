"""Selected saved-arithmetic functions; no producer entry points."""

from __future__ import annotations

from fractions import Fraction as F

from math import isqrt

BITS=256

SCALE=1<<BITS

ZERO=(F(0),F(0))

ONE=(F(1),F(1))

NAMES=('Hbb','Haa','Hperp')

MUS={'Hbb':F(1,10),'Haa':F(1,50),'Hperp':F(1,200)}

def require(condition,message):
    if not condition:raise ValueError(message)

def dyadic(x):
    require(type(x['exponent_of_two']) is int,'integer dyadic exponent')
    return F(int(x['mantissa']))*F(2)**x['exponent_of_two']

def iv(x):
    if isinstance(x,dict):
        a,b=dyadic(x['lower_exact_dyadic']),dyadic(x['upper_exact_dyadic'])
    elif isinstance(x,tuple):return x
    else:a=b=F(x)
    require(a<=b,'ordered interval')
    return a,b

def down(x):return F((x.numerator*SCALE)//x.denominator,SCALE)

def up(x):return -down(-x)

def rounded(a,b):return down(a),up(b)

def add(a,b):return rounded(a[0]+b[0],a[1]+b[1])

def neg(a):return -a[1],-a[0]

def sub(a,b):return add(a,neg(b))

def mul(a,b):
    v=[x*y for x in a for y in b]
    return rounded(min(v),max(v))

def div(a,b):
    require(b[0]>0 or b[1]<0,'division interval excludes zero')
    return mul(a,(1/b[1],1/b[0]))

def ab(a):return max(abs(a[0]),abs(a[1]))

def expand(a,e):return rounded(a[0]-e,a[1]+e)

def sumiv(xs):
    out=ZERO
    for x in xs:out=add(out,x)
    return out

def contains(outer,inner):return outer[0]<=inner[0] and inner[1]<=outer[1]

def overlap(a,b):return a[0]<=b[1] and b[0]<=a[1]

def intersect(a,b):
    require(overlap(a,b),'inconsistent purported enclosures')
    return max(a[0],b[0]),min(a[1],b[1])

def sqrt_bounds(q):
    require(q>=0,'nonnegative rational sqrt')
    scaled_n=q.numerator*SCALE*SCALE
    k=isqrt(scaled_n//q.denominator)
    require(k*k*q.denominator<=scaled_n,'isqrt lower square audit')
    j=k if k*k*q.denominator==scaled_n else k+1
    require(j*j*q.denominator>=scaled_n,'isqrt upper square audit')
    return F(k,SCALE),F(j,SCALE)

def sqrt_iv(a):
    require(a[0]>=0,'nonnegative sqrt interval')
    return sqrt_bounds(a[0])[0],sqrt_bounds(a[1])[1]

def out(a):return dict(lower_exact=str(a[0]),upper_exact=str(a[1]),midpoint_display=float((a[0]+a[1])/2))

def point(q):return F(q),F(q)

def readmat(A):return [[iv(x) for x in row] for row in A]

def transpose(A):return [list(x) for x in zip(*A)]

def matmul(A,B):return [[sumiv(mul(A[i][k],B[k][j]) for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def matvec(A,b):return [sumiv(mul(x,y) for x,y in zip(row,b)) for row in A]

def zeros(n=8,m=8):return [[ZERO for _ in range(m)] for _ in range(n)]

def householder(a):
    norm=sqrt_iv(sumiv(mul(x,x) for x in a))
    v=[div(x,norm) for x in a];v[0]=add(v[0],ONE)
    den=sumiv(mul(x,x) for x in v)
    return [[sub(point(int(i==j)),div(mul(point(2),mul(v[i],v[j])),den)) for j in range(1,8)] for i in range(8)]

def ldl(A,shift):
    n=len(A);L=zeros(n,n);D=[]
    for i in range(n):
        L[i][i]=ONE
        pivot=sub(sub(A[i][i],shift),sumiv(mul(mul(L[i][k],L[i][k]),D[k]) for k in range(i)))
        D.append(pivot)
        if pivot[0]<=0:return dict(passed=False,failed_pivot=i,pivots=[out(x) for x in D])
        for j in range(i+1,n):
            numerator=sub(A[j][i],sumiv(mul(mul(L[j][k],L[i][k]),D[k]) for k in range(i)))
            L[j][i]=div(numerator,pivot)
    return dict(passed=True,failed_pivot=None,pivots=[out(x) for x in D])

def read_conditional(blocks):
    rows={};cfg=blocks[0]['config'];signature=blocks[0]['model_signature']
    expected=int((F(cfg['vmax'])-F(cfg['vmin']))/F(cfg['hv']))+1
    for block in blocks:
        require(block['status']=='completed' and block['partial_conditional_row'] is None,'complete block')
        require(block['config']==cfg and block['model_signature']==signature,'same conditional box/config')
        for row in block['conditional_rows']:
            j=row['label_index'];require(j not in rows and 0<=j<128,'unique complement representative')
            require(row['status']=='completed' and row['outer_nodes_completed']==expected and row['partial_current_inner_factors']==0,'complete conditional row')
            require(row['label_bits']==[(j>>(7-i))&1 for i in range(8)],'label order')
            rows[j]=dict(g=[iv(x) for x in row['conditional_gradient_finite']],
                         **{name:readmat(row['matrix_blocks'][name]) for name in NAMES})
    require(sorted(rows)==list(range(128)),'all 128 representatives')
    return rows

def audit_teacher(teacher,finite):
    q=[iv(x) for x in teacher['q_true_component_intervals']]
    p=[iv(x) for x in teacher['prime_true_component_intervals']]
    eq=iv(teacher['analytic_L1_errors']['q'])[1];ep=iv(teacher['analytic_L1_errors']['prime'])[1]
    for original,saved,error in ((finite['q_finite'],q,eq),(finite['prime_finite'],p,ep)):
        require(len(original)==len(saved)==256,'complete teacher arrays')
        for x,y in zip(original,saved):require(contains(y,expand(iv(x),error)),'teacher component error transfer')
    require(sumiv(q)[0]<=1<=sumiv(q)[1] and sumiv(p)[0]<=0<=sumiv(p)[1],'teacher mass compatibility')
    require(F(teacher['sum_qprime_exact'])==0,'analytic teacher derivative mass declaration')
    require(iv(teacher['qsecond_global_L1_cap'])[1]<F('.442323'),'second derivative prescribed cap')
    return q,p,dict(component_transfers_checked=512,teacher_alias_bounds_recomputed=False,
                    true_component_enclosures_and_zero_mass_declaration_checked=True)

def signed_rebuild(rows,q,p,h,cert):
    G=[ZERO]*8;H0={n:zeros() for n in NAMES};H1={n:zeros() for n in NAMES}
    for label in range(128):
        w0=add(q[label],q[255-label]);w1=add(p[label],p[255-label])
        for i in range(8):
            G[i]=add(G[i],mul(w1,rows[label]['g'][i]))
            for j in range(8):
                for name in NAMES:
                    H0[name][i][j]=add(H0[name][i][j],mul(w0,rows[label][name][i][j]))
                    H1[name][i][j]=add(H1[name][i][j],mul(w1,rows[label][name][i][j]))
    H={name:[[add(H0[name][i][j],mul((-h,h),H1[name][i][j])) for j in range(8)] for i in range(8)] for name in NAMES}
    require(all(contains(iv(s),a) for s,a in zip(cert['signed_combinations']['G1_box'],G)),'saved signed G1 enclosure')
    for key,data in (('H0',H0),('H1',H1),('Hlambda_finite',H)):
        for name in NAMES:
            saved=readmat(cert['signed_combinations'][key][name])
            require(all(contains(saved[i][j],data[name][i][j]) for i in range(8) for j in range(8)),'saved signed matrix enclosure '+key+'/'+name)
    return G,H,dict(conditional_labels=128,unfolded_teacher_labels=256,gradient_coordinates=8,
                   matrix_entries_checked=3*3*64,signed_weights_preserved=True)

def audit_root(cert,rows,G,H,teacher,bounds):
    h=F(cert['h_exact']);r=list(map(F,cert['radii_exact']));A=[[point(F(x)) for x in row] for row in cert['A_exact']]
    q1=iv(teacher['prime_true_L1'])[1];tail=F('.2211615')*h*h
    epsg=iv(bounds['bounds']['uniform_gradient_L2'])[1];epsH=iv(bounds['bounds']['uniform_Hba_operator'])[1]
    epsP=iv(bounds['bounds']['uniform_Hperp_operator'])[1]
    EH=epsH*(1+h*q1)+F('146.25')*tail;EP=epsP*(1+h*q1)+F('8.25')*tail
    require(EH<=iv(cert['analytic_errors']['Hba_total_operator'])[1] and EP<=iv(cert['analytic_errors']['Hperp_total_operator'])[1],'Hessian transfer upper caps')
    oldg=F('.0000000007572');olddist=F('.00000001805')
    require(iv(cert['contraction']['old_gradient_L2_cap'])[1]>=oldg,'old gradient cap retained')
    require(iv(cert['contraction']['old_root_distance_cap'])[1]>=olddist,'old root distance cap retained')
    rn=[sqrt_bounds(sum(x[0]*x[0] for x in row))[1] for row in A];rnorm=sqrt_bounds(sum(x*x for x in r))[1]
    AG=matvec(A,G);AH=matmul(A,H['Haa']);transformed=[matvec(A,rows[y]['g']) for y in range(128)]
    details=[];proof=ldl(A,ZERO)
    for i in range(8):
        transfer=cert['teacher_gradient_zero_mass_transfer']['rows'][i]
        center=F(transfer['centre_exact'])
        centered=max(ab(sub(v[i],point(center))) for v in transformed)+rn[i]*epsg
        legacy=8*rn[i];chosen=min(centered,legacy)
        require(chosen<=iv(transfer['selected_true_cap'])[1],'saved zero-mass row cap')
        y=oldg*rn[i]+h*ab(AG[i])+h*rn[i]*epsg*q1+tail*chosen
        zfinite=sum(ab(sub(point(int(i==j)),AH[i][j]))*r[j] for j in range(8))
        z=zfinite+rn[i]*EH*rnorm
        oldrow=cert['contraction']['rows'][i]
        require(y<=iv(oldrow['Y'])[1] and z<=iv(oldrow['Z'])[1],'saved Y/Z upper encloses independent bound')
        details.append(dict(coordinate=i+1,Y_upper_exact=str(y),Z_upper_exact=str(z),
            independent_inclusion_margin_exact=str(r[i]-y-z),independent_contraction_ratio_exact=str(z/r[i]),
            zero_mass_cap_upper_exact=str(chosen),legacy_cap_upper_exact=str(legacy),
            self_mapping=y+z<r[i],contraction=z<r[i]))
    passed=proof['passed'] and all(x['self_mapping'] and x['contraction'] for x in details) and olddist<min(r)
    require(passed==cert['contraction']['passed'],'h001 root pass status independently reproduced')
    return dict(passed=passed,A_positive_definite=proof,rows=details,
                old_root_inside=olddist<min(r),teacher_remainder_L1_exact=str(tail),
                Hba_transfer_error_upper_exact=str(EH),Hperp_transfer_error_upper_exact=str(EP))

def audit_risk(raw,cert,q,p,h):
    require(raw['status']=='complete' and raw['composition']['own_target_numerator_sum_zero'],'complete own-target risk input')
    eQ=iv(raw['analytic_errors']['Q'])[1];eN=iv(raw['analytic_errors']['N'])[1]
    Q=[expand(iv(x),eQ) for x in raw['Q_finite']];N=[expand(iv(x),eN) for x in raw['N_finite']]
    require(all(x[0]>0 for x in Q),'strictly positive independent Q')
    rho=[div(n,qq) for n,qq in zip(N,Q)]
    for key,reconstructed in (('Q_true_intervals',Q),('N_true_intervals',N),('rho_true_intervals',rho)):
        require(all(contains(iv(saved),x) for saved,x in zip(raw['composition'][key],reconstructed)),'risk finite/error/ratio reconstruction '+key)
    original=sumiv(mul(qy,ry) for qy,ry in zip(q,rho))
    centered=sumiv(mul(sub(qy,Qy),ry) for qy,Qy,ry in zip(q,Q,rho))
    R0=intersect(original,centered);R1=sumiv(mul(py,ry) for py,ry in zip(p,rho))
    # Use the producer's saved fixed B0 solely as an exact chosen center. Any
    # such real number is valid under the analytic remainder mass-zero identity.
    B0=sum(iv(cert['risk']['centering_B0']))/2
    radius=max(ab(sub(x,point(B0))) for x in rho)
    tail=F('.2211615')*h*h*radius
    total=expand(add(R0,mul((-h,h),R1)),tail)
    require(total[1]<0,'independent strict negative risk curvature')
    return dict(passed=True,label_components=256,Q_positive=True,
        finite_to_true_transfers_checked=512,quotients_independently_rebuilt=256,
        R0_original=out(original),R0_centered=out(centered),R0_intersection=out(R0),R1=out(R1),
        Taylor_remainder_upper_exact=str(tail),true_curvature=out(total),
        saved_curvature=cert['risk']['true_curvature'],
        analytic_own_target_sum_N_zero_assumed_from_bound_proof=True,
        numerical_zero_sum_used_as_proof=False)

def audit_spectrum(cert,refinement):
    U=householder([point(F(x)) for x in cert['a0_exact']]);blocks={}
    gram=matmul(transpose(U),U)
    require(all(contains(gram[i][j],point(int(i==j))) for i in range(7) for j in range(7)),'Householder identity numerical consistency')
    for name in NAMES:
        H=readmat(cert['signed_combinations']['Hlambda_finite'][name]);new=refinement['blocks'][name]
        mid=[[(x[0]+x[1])/2 for x in row] for row in H]
        radius2=sum(((x[1]-x[0])/2)**2 for row in H for x in row)
        saved_mid=[[dyadic(x) for x in row] for row in new['midpoint_exact_dyadic']]
        require(mid==saved_mid,'spectral exact midpoint reconstruction')
        radius=dyadic(new['rho_F_upper_exact_dyadic'])
        require(radius>=0 and radius2<=radius*radius,'independent spectral radius squared bound')
        require(radius2==F(new['Fraction_radius_audit']['sum_halfwidth_squares_exact']),'saved squared radius equality')
        error=iv(cert['analytic_errors']['Hperp_total_operator' if name=='Hperp' else 'Hba_total_operator'])[1]
        shift=MUS[name]+error+radius
        require(shift==F(new['total_shift_exact']) and F(new['target_mu_exact'])==MUS[name],'same spectral shift and mu')
        pointmatrix=[[point(x) for x in row] for row in mid]
        checked=matmul(matmul(transpose(U),pointmatrix),U) if name=='Hperp' else pointmatrix
        saved_checked=readmat(new['checked_midpoint_matrix'])
        require(all(contains(saved_checked[i][j],checked[i][j]) for i in range(len(checked)) for j in range(len(checked))),'spectral point projection enclosure')
        proof=ldl(checked,point(shift))
        require(proof['passed']==new['ldl']['passed'],'independent refined LDL status')
        originalmatrix=readmat(cert['transverse_slice_matrix']) if name=='Hperp' else H
        original=ldl(originalmatrix,point(MUS[name]+error))
        oldkey='transverse_slice' if name=='Hperp' else name
        if cert['stability'][oldkey]['passed']:require(original['passed'],'original positive LDL remains provable')
        blocks[name]=dict(mu_exact=str(MUS[name]),rho_squared_exact=str(radius2),rho_upper_exact=str(radius),
            square_margin_exact=str(radius*radius-radius2),shift_exact=str(shift),
            midpoint_entries_checked=64,projected_dimension=len(checked),refined_LDL=proof,
            original_LDL_from_saved_enclosure=original,original_recorded_passed=cert['stability'][oldkey]['passed'])
    core=cert['contraction']['passed'] and cert['risk']['passed'] and all(x['refined_LDL']['passed'] for x in blocks.values())
    require(core==refinement['core_passed'],'same refined core status')
    require(refinement['original_stability']==cert['stability'],'old failures retained')
    return dict(h_exact=cert['h_exact'],blocks=blocks,core_passed=core,
        original_root_passed=cert['contraction']['passed'],original_risk_passed=cert['risk']['passed'],
        uses_original_preprojection_radius=True,Householder_orthogonality_analytic_proof_not_replaced=True)
