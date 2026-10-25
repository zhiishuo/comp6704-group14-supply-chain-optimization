import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from model import generate,highs,metrics
from solvers import METHODS,pdhg


def main():
    checks=[]
    for seed in [0,1,2]:
        p=generate(2,3,4,seed); reference=highs(p); f=p.c@reference['x']
        for name,fn in METHODS.items():
            r=fn(p); met=metrics(p,r['x'],r['y'])
            assert np.isfinite(met['kkt'])
            assert met['dual_bound']<=f+1e-7,(seed,name,met)
            if r['status']=='converged':
                assert met['kkt']<=1e-6
                assert abs(p.c@r['x']-f)/(1+abs(f))<2e-5
            else: assert name=='PR-PDHG' and r['status'] in ['iteration_limit','time_limit']
            checks.append(dict(seed=seed,method=name,status=r['status'],iterations=r['iterations'],kkt=met['kkt']))
        assert abs(p.emission@reference['x']-p.b[-1])/(1+p.b[-1])<1e-7
    p=generate(2,3,4,0)
    r=pdhg(p,max_iter=1); assert r['status']=='iteration_limit'
    # Arbitrary valid dual multipliers must be lower bounds.
    f=p.c@highs(p)['x']
    for seed in range(10):
        y=np.random.default_rng(seed).normal(size=len(p.b)); y[p.neq:]=-abs(y[p.neq:])
        assert metrics(p,np.zeros_like(p.c),y)['dual_bound']<=f+1e-9
    target=Path(__file__).absolute().parent/'results'; target.mkdir(exist_ok=True)
    (target/'tests.json').write_text(json.dumps({'status':'PASS','solver_seed_checks':checks,
        'additional_checks':['carbon budget active','budget exhaustion','ten arbitrary dual lower bounds']},indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
