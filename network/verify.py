"""Independent original-unit certificate audit; uses scalar sums, no solver metric."""
from pathlib import Path
import math,csv,json
import numpy as np
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits
from model import load,generate

ROOT=Path(__file__).absolute().parent; OUT=ROOT/'results'


def independent(p,x,y):
    activity=[]
    for i in range(len(p.b)):
        lo,hi=p.A.indptr[i:i+2]
        activity.append(math.fsum(float(v)*float(x[j]) for v,j in zip(p.A.data[lo:hi],p.A.indices[lo:hi])))
    violations=[(abs(a-float(b)) if i<p.neq else max(a-float(b),0))/(1+abs(float(b))) for i,(a,b) in enumerate(zip(activity,p.b))]
    rp=max(violations+[max(0,-float(xx),float(xx-uu))/(1+float(uu)) for xx,uu in zip(x,p.u)])
    rd=max([0]+[float(v) for v in y[p.neq:]])/(1+max(abs(float(v)) for v in y))
    yf=y.copy(); yf[p.neq:]=np.minimum(yf[p.neq:],0)
    at=p.A.tocsc(); reduced=[]
    for j in range(len(x)):
        lo,hi=at.indptr[j:j+2]
        reduced.append(float(p.c[j])-math.fsum(float(a)*float(yf[i]) for a,i in zip(at.data[lo:hi],at.indices[lo:hi])))
    objective=math.fsum(float(a)*float(b) for a,b in zip(p.c,x))
    lower=math.fsum(float(a)*float(b) for a,b in zip(p.b,yf))+math.fsum(float(u)*min(r,0) for u,r in zip(p.u,reduced))
    gap=abs(objective-lower)/(1+abs(objective)+abs(lower))
    return dict(objective=objective,dual_bound=lower,primal_residual=rp,dual_sign_residual=rd,gap=gap,kkt=max(rp,rd,gap))


def main():
    rows=list(csv.DictReader((OUT/'comparison.csv').open())); details=[]; oracle={}; successes=0
    for row in rows:
        p=load(OUT/'instances'/f'{row["case"]}.npz')
        for rep in range(3):
            z=np.load(OUT/'solutions'/f'{row["case"]}_{row["method"]}_r{rep}.npz')
            met=independent(p,z['x'],z['y'])
            assert met['dual_bound']<=float(row['reference_objective'])+1e-7*(1+abs(float(row['reference_objective'])))
            if rep==0:
                for key in met: assert abs(met[key]-float(row[key]))<1e-8*(1+abs(met[key])),(row['case'],row['method'],key)
                if row['status']=='converged': assert met['kkt']<=1.00001e-6
            successes+=int(met['kkt']<=1e-6)
            details.append(dict(case=row['case'],method=row['method'],repeat=rep,**met))
        if row['case'] not in oracle:
            r=linprog(p.c,A_eq=p.A[:p.neq],b_eq=p.b[:p.neq],A_ub=p.A[p.neq:],b_ub=p.b[p.neq:],
                bounds=np.column_stack([np.zeros_like(p.u),p.u]),method='highs-ipm',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'ipm_optimality_tolerance':1e-10})
            assert r.success
            assert abs(r.fun-float(row['reference_objective']))<1e-7*(1+abs(r.fun))
            oracle[row['case']]=r.fun
    ab=list(csv.DictReader((OUT/'ablation.csv').open()))
    for row in ab:
        p=generate(10,15,20,int(row['seed'])); z=np.load(OUT/'solutions'/f'{row["case"]}.npz')
        met=independent(p,z['x'],z['y']); assert abs(met['kkt']-float(row['kkt']))<1e-8
    frontier=list(csv.DictReader((OUT/'carbon_frontier.csv').open()))
    for row in frontier:
        p=generate(10,15,20,int(row['seed']),float(row['beta'])); z=np.load(OUT/'solutions'/f'{row["case"]}.npz')
        met=independent(p,z['x'],z['y']); assert met['kkt']<1e-8
    for seed in sorted(set(r['seed'] for r in frontier)):
        s=sorted([r for r in frontier if r['seed']==seed],key=lambda r:float(r['beta']))
        assert all(float(s[j+1]['cost'])<=float(s[j]['cost'])+1e-7 for j in range(len(s)-1))
    info=dict(status='PASS',main_certificates=len(details),certified_repetitions=successes,
        reference_crosschecks=len(oracle),ablation_certificates=len(ab),carbon_certificates=len(frontier),
        max_carbon_derivative_error=max(float(r['derivative_error']) for r in frontier),details=details)
    (OUT/'verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='details'},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
