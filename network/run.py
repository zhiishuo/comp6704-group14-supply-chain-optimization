"""Frozen two-echelon benchmark, ablations, carbon frontier and constraint audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
from time import perf_counter
import csv,json,sys,platform,hashlib
import numpy as np
import scipy,ortools
from threadpoolctl import threadpool_limits,threadpool_info
from model import generate,save,highs,metrics
from solvers import METHODS,pdhg

ROOT=Path(__file__).absolute().parent; OUT=ROOT/'results'; CFG=json.loads((ROOT/'protocol.json').read_text())


def csvwrite(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def audit(p,x,y):
    a=p.A@x; k,n,m=p.k,p.n,p.m
    ranges={'warehouse_balance':(0,k),'demand':(k,k+n),'factory_capacity':(k+n,k+n+m),
            'warehouse_capacity':(k+n+m,k+n+m+k),'carbon':(len(p.b)-1,len(p.b))}
    detail={}
    for name,(lo,hi) in ranges.items():
        slack=p.b[lo:hi]-a[lo:hi]
        violation=np.abs(slack) if lo<p.neq else np.maximum(-slack,0)
        detail[name]={'max_normalized_violation':float(np.max(violation/(1+np.abs(p.b[lo:hi])))),
                     'active_count':int(np.count_nonzero(np.abs(slack)<=1e-6*(1+np.abs(p.b[lo:hi])))),
                     'total':hi-lo,'dual_min':float(min(y[lo:hi])),'dual_max':float(max(y[lo:hi]))}
    detail['route_capacity']={'max_normalized_violation':float(np.max(np.maximum(x-p.u,0)/(1+p.u))),
        'active_count':int(np.count_nonzero(np.abs(x-p.u)<=1e-6*(1+p.u))),'total':len(x)}
    return detail


def main():
    OUT.mkdir(exist_ok=True)
    for d in ['instances','solutions']: (OUT/d).mkdir(exist_ok=True)
    (OUT/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        ortools=ortools.__version__,platform=platform.platform(),processor=platform.processor(),blas=threadpool_info()),indent=2),encoding='utf-8')
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    source['protocol.json']=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    (OUT/'source_hashes.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    warm=generate(2,3,4,0)
    for fn in METHODS.values(): fn(warm)
    rows=[]; timing=[]; histories={}; index={}; rng=np.random.default_rng(7129)
    cases=[]
    for size in CFG['small_sizes']:
        cases += [('small',size,s) for s in CFG['small_seeds']]
    for size in CFG['large_sizes']:
        cases += [('large',size,s) for s in CFG['large_seeds']]
    cases += [('stress',CFG['stress_size'],s) for s in CFG['stress_seeds']]
    for cohort,size,seed in cases:
        case='m%d_k%d_n%d_s%d'%(*size,seed); p=generate(*size,seed,CFG['carbon_beta']); save(p,OUT/'instances'/f'{case}.npz')
        index[case]=dict(cohort=cohort,size=size,**p.meta)
        ref=highs(p); objective=float(p.c@ref['x'])
        names=list(METHODS) if cohort=='small' else ['Sparse-IPM','PR-PDHG','PDLP']
        names+=['HiGHS']; fns={**METHODS,'HiGHS':highs}; res={name:[] for name in names}
        for repeat in range(CFG['repeats']):
            for name in rng.permutation(names):
                ans=fns[name](p); res[name].append(ans)
                timing.append(dict(case=case,cohort=cohort,method=name,repeat=repeat,seconds=ans['seconds'],
                    setup_seconds=ans['setup_seconds'],iterations=ans['iterations'],status=ans['status']))
        for name,answers in res.items():
            ans=answers[0]; t=np.array([a['seconds'] for a in answers]); met=metrics(p,ans['x'],ans['y'])
            # Time caps may alter final iterations; certify and archive every repetition.
            repeat_success=0
            for j,a in enumerate(answers):
                mm=metrics(p,a['x'],a['y']); repeat_success+=int(mm['kkt']<=CFG['tolerance'])
                np.savez_compressed(OUT/'solutions'/f'{case}_{name}_r{j}.npz',x=a['x'],y=a['y'])
            row=dict(case=case,cohort=cohort,m=p.m,k=p.k,n=p.n,variables=len(p.c),constraints=len(p.b),nnz=p.A.nnz,
                method=name,status=ans['status'],certified_repeats=repeat_success,iterations=ans['iterations'],phase1=ans['phase1'],
                restarts=ans['restarts'],time_median_ms=float(np.median(t)*1000),time_mean_ms=float(t.mean()*1000),
                time_sd_ms=float(t.std(ddof=1)*1000),setup_median_ms=float(np.median([a['setup_seconds'] for a in answers])*1000),
                objective_error=abs(met['objective']-objective)/(1+abs(objective)),reference_objective=objective,**met)
            rows.append(row); histories[case+'/'+name]=ans['history']
            print(case,name,ans['status'],ans['iterations'],f'{row["time_median_ms"]:.1f}ms',f'R={met["kkt"]:.2e}',flush=True)
        csvwrite('comparison.csv',rows); csvwrite('timing_repetitions.csv',timing)
        (OUT/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
        (OUT/'histories.json').write_text(json.dumps(histories),encoding='utf-8')
    ab=[]
    for seed in CFG['ablation_seeds']:
        p=generate(*CFG['ablation_size'],seed)
        for name,pre,restart in [('Scalar-PDHG',False,False),('P-PDHG',True,False),('PR-PDHG',True,True)]:
            a=pdhg(p,max_iter=CFG['ablation_iterations'],time_limit=CFG['ablation_time_seconds'],precondition=pre,restart=restart)
            case=f'ablation_s{seed}_{name}'
            np.savez_compressed(OUT/'solutions'/f'{case}.npz',x=a['x'],y=a['y'])
            ab.append(dict(case=case,seed=seed,method=name,status=a['status'],iterations=a['iterations'],seconds=a['seconds'],
                restarts=a['restarts'],**metrics(p,a['x'],a['y'])))
        a=pdhg(p,max_iter=CFG['ablation_iterations'],time_limit=CFG['ablation_time_seconds'],row_scale=True)
        case=f'ablation_s{seed}_RHS-PR-PDHG'
        np.savez_compressed(OUT/'solutions'/f'{case}.npz',x=a['x'],y=a['y'])
        ab.append(dict(case=case,seed=seed,method='RHS-PR-PDHG',status=a['status'],iterations=a['iterations'],seconds=a['seconds'],
                      restarts=a['restarts'],**metrics(p,a['x'],a['y'])))
        print('ablation',seed,flush=True)
    csvwrite('ablation.csv',ab)
    frontier=[]; constraint_rows=[]
    for seed in CFG['ablation_seeds']:
        base=generate(*CFG['ablation_size'],seed)
        width=base.meta['unconstrained_cost_emission']-base.meta['min_emission']
        for beta in CFG['carbon_betas']:
            p=generate(*CFG['ablation_size'],seed,beta); a=highs(p); cost=float(p.c@a['x'])
            deriv=float(a['y'][-1]*width); old=float(p.b[-1]); p.b[-1]+=1e-5*width; near=highs(p); p.b[-1]=old
            fd=float((p.c@near['x']-cost)/1e-5); case=f'carbon_s{seed}_b{beta}'
            np.savez_compressed(OUT/'solutions'/f'{case}.npz',x=a['x'],y=a['y'])
            frontier.append(dict(case=case,seed=seed,beta=beta,cost=cost,emissions=float(p.emission@a['x']),
                budget=old,unconstrained_cost=p.meta['unconstrained_cost'],carbon_price=-a['y'][-1],derivative=deriv,
                derivative_fd=fd,derivative_error=abs(deriv-fd),**{'kkt':metrics(p,a['x'],a['y'])['kkt']}))
        base=generate(*CFG['ablation_size'],seed); a=highs(base); basecost=float(base.c@a['x'])
        for name,lo,hi in [('factory',base.neq,base.neq+base.m),
                           ('warehouse',base.neq+base.m,base.neq+base.m+base.k),('carbon',len(base.b)-1,len(base.b))]:
            saved=base.b[lo:hi].copy(); base.b[lo:hi]*=100
            altered=highs(base); newcost=float(base.c@altered['x']); base.b[lo:hi]=saved
            constraint_rows.append(dict(seed=seed,relaxed=name,base_cost=basecost,relaxed_cost=newcost,saving_pct=100*(1-newcost/basecost)))
        if seed==CFG['ablation_seeds'][0]:
            (OUT/'constraint_audit.json').write_text(json.dumps(audit(base,a['x'],a['y']),indent=2),encoding='utf-8')
        print('carbon frontier',seed,flush=True)
    csvwrite('carbon_frontier.csv',frontier); csvwrite('constraint_ablation.csv',constraint_rows)
    print('ALL NETWORK EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1): main()
