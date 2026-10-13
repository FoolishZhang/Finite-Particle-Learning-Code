"""Post-CAL local null geometry diagnostic; no new samples or primary gates."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/calibration_precision/NULL_CURVATURE'
MODEL=ROOT/'artifacts/calibration_precision/CAL/FULL/MODEL.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def run():
    OUT.mkdir(exist_ok=True)
    write(OUT/'PROTOCOL.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),full_model_sha256=sha(MODEL),stage='Post-CAL exploratory diagnostic, before root reads new TRAIN risk results; outside original five-gate family',motivation='CAL b(0)*rprime(0) versus 2*local coefficient suggested null raw-curvature sign change across n; inspect directly stored U_prime without changing training or selecting new inputs.',scope='All four cases, U_prime not-a-knot derivative at zero and fixed symmetric differences h=.03,.06,.12; per-stream SE, no success gates/CI or method tuning.',new_random_samples=0,new_model_integrals=0,does_not_prove_global_minimum=True))
    m=json.loads(MODEL.read_text('utf8'));rows=[]
    for c in m['cases']:
        x=np.array(c['rho101']);u=np.array(c['per_stream']['U_prime']);v=CubicSpline(x,u,axis=1)(0.,1)
        diffs={}
        for h in (.03,.06,.12):
            ip=int(np.argmin(abs(x-h)));im=int(np.argmin(abs(x+h)));assert max(abs(x[ip]-h),abs(x[im]+h))<1e-14
            vals=(u[:,ip]-u[:,im])/(x[ip]-x[im]);diffs[str(h)]=dict(mean=float(vals.mean()),SE=float(vals.std(ddof=1)/4))
        rows.append(dict(n=c['n'],K=c['K'],spline_U_second_zero=float(v.mean()),stream_SE=float(v.std(ddof=1)/4),stream_values=v.tolist(),central_differences=diffs))
    write(OUT/'RESULTS.json',dict(source_sha256=sha(__file__),protocol_sha256=sha(OUT/'PROTOCOL.json'),model_sha256=sha(MODEL),rows=rows,new_random_samples=0,scope='Numerical local curvature only. U even implies stationary zero in the exact null mean target; positive/negative second derivative suggests local min/max, not global certification. SE is not a simultaneous confidence band.'))
    print(json.dumps(rows))
if __name__=='__main__':run()
