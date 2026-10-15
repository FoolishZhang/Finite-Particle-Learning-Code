"""Euclidean projection onto unit diagonal matrices with a spectral floor.

Dykstra-corrected alternating projections; no optimizer/momentum state.
Reference: Higham (2002), IMA J Numerical Analysis 22, 329-343,
https://doi.org/10.1093/imanum/22.3.329 . This is a numerical projection,
not an interval certificate. Off-diagonal Euclidean and matrix Frobenius
norms differ by a constant factor and yield the same projection here.
"""
import numpy as np
import signal_proposal_core as base

def project_coordinates(c, delta=.1, tol=1e-12, max_iter=2000):
    A=base.correlation_matrix(c)
    if not 0<delta<1 or tol<=0:raise ValueError('Invalid projection tolerance/floor')
    lam=float(np.linalg.eigvalsh(A)[0])
    if lam>=delta:return {'coordinates':np.asarray(c).copy(),'iterations':0,'lambda_min':lam,'residual':0.,'roundoff_retraction':1.}
    Y=A.copy();X=A.copy();correction=np.zeros((3,3))
    for it in range(1,max_iter+1):
        oldX=X.copy();oldY=Y.copy();R=Y-correction
        values,vectors=np.linalg.eigh(R)
        X=(vectors*np.maximum(values,delta))@vectors.T;correction=X-R
        Y=X.copy();np.fill_diagonal(Y,1.);Y=(Y+Y.T)/2
        residual=max(np.linalg.norm(X-oldX,'fro'),np.linalg.norm(Y-oldY,'fro'),np.linalg.norm(Y-X,'fro'))
        if residual<=tol*max(1.,np.linalg.norm(Y,'fro')):break
    else:raise RuntimeError('Dykstra projection did not reach tolerance')
    # Restore floating spectral feasibility at a displacement of roundoff size.
    safe=base.radial_retract(Y,delta)
    if np.linalg.norm(safe['matrix']-Y,'fro')>50*tol:raise ArithmeticError('Projection feasibility correction exceeds tolerance allowance')
    return {'coordinates':base.correlation_coordinates(safe['matrix']),'iterations':it,
        'lambda_min':safe['lambda_min'],'residual':residual,'roundoff_retraction':safe['t']}

if __name__=='__main__':
    from scipy.optimize import minimize
    records=[]
    for c in [[.2,-.1,.15],[1.,1.,1.],[1.,1.,-1.],[.9,.8,-.2],[-.95,.71,.63]]:
        x=project_coordinates(c);a=np.asarray(c)
        if np.all(np.abs(a)==1):
            expected=a*(.9 if np.prod(a)>0 else .45)
            np.testing.assert_allclose(x['coordinates'],expected,atol=3e-12,rtol=0)
            records.append({'input':c,'iterations':x['iterations'],'analytic_symmetry_solution_error':float(np.linalg.norm(x['coordinates']-expected))})
            continue
        d=.9
        opt=minimize(lambda u:.5*np.sum((u-a)**2),np.zeros(3),jac=lambda u:u-a,bounds=[(-d,d)]*3,
            constraints=[{'type':'ineq','fun':lambda u:d**3+2*np.prod(u)-d*np.sum(u*u),
                'jac':lambda u:2*np.array([u[1]*u[2],u[0]*u[2],u[0]*u[1]])-2*d*u}],
            method='SLSQP',options={'ftol':1e-13,'maxiter':1000})
        assert opt.success,opt.message
        assert np.linalg.norm(opt.x-x['coordinates'])<3e-6
        assert np.linalg.eigvalsh(base.correlation_matrix(x['coordinates']))[0]>=.1
        np.testing.assert_allclose(project_coordinates(x['coordinates'])['coordinates'],x['coordinates'],atol=2e-12,rtol=0)
        records.append({'input':c,'iterations':x['iterations'],'independent_SLSQP_distance':float(np.linalg.norm(opt.x-x['coordinates']))})
    np.testing.assert_allclose(project_coordinates([1,1,-1])['coordinates'],[.45,.45,-.45],atol=3e-12)
    import json
    print(json.dumps({'manufactured_only':True,'passed':True,'cases':records}))
