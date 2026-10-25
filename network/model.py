"""Two-echelon continuous supply-chain LP with six constraint families."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


@dataclass
class Network:
    A: sp.csr_matrix
    b: np.ndarray
    c: np.ndarray
    u: np.ndarray
    neq: int
    m: int
    k: int
    n: int
    emission: np.ndarray
    witness: np.ndarray
    meta: dict


def highs(p, objective=None):
    t=perf_counter()
    r=linprog(p.c if objective is None else objective,A_eq=p.A[:p.neq],b_eq=p.b[:p.neq],
        A_ub=p.A[p.neq:],b_ub=p.b[p.neq:],bounds=np.column_stack([np.zeros_like(p.u),p.u]),
        method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not r.success: raise RuntimeError(r.message)
    return dict(x=r.x,y=np.r_[r.eqlin.marginals,r.ineqlin.marginals],iterations=int(r.nit),
                seconds=perf_counter()-t,setup_seconds=0.,status='converged',history=[],phase1=0,restarts=0)


def generate(m,k,n,seed,beta=.5):
    """The carbon budget interpolates known feasible min-emission and min-cost plans.

    The two auxiliary HiGHS solves are data construction, outside timing. Their
    solutions never initialize any compared method. beta controls carbon strictness.
    """
    rng=np.random.default_rng(seed); nf=m*k; nd=k*n; e=nf+nd
    posf=rng.uniform(0,1,(m,2)); posw=rng.uniform(0,1,(k,2)); posd=rng.uniform(0,1,(n,2))
    distance=np.r_[np.linalg.norm(posf[:,None]-posw[None,:],axis=2).ravel(),
                   np.linalg.norm(posw[:,None]-posd[None,:],axis=2).ravel()]
    cleaner=rng.uniform(0,1,e)
    c=.2+distance*(.7+.9*cleaner)+rng.uniform(0,.03,e)
    handling=rng.uniform(.05,.15,k); c[:nf]+=np.tile(handling,m)
    emission=.1+distance*(1.7-1.1*cleaner)
    f=rng.uniform(.5,1.5,(m,k)); z=rng.uniform(.5,1.5,(k,n))
    z*=f.sum(axis=0)[:,None]/z.sum(axis=1)[:,None]
    witness=np.r_[f.ravel(),z.ravel()]; u=2*witness
    # Row order: warehouse equality, demand equality, factory supply,
    # warehouse throughput, total carbon budget.
    rr=[]; cc=[]; vv=[]
    for i in range(m):
        for w in range(k):
            j=i*k+w
            rr.extend([w,k+n+i,k+n+m+w]); cc.extend([j]*3); vv.extend([1,1,1])
    for w in range(k):
        for j in range(n):
            idx=nf+w*n+j
            rr.extend([w,k+j]); cc.extend([idx]*2); vv.extend([-1,1])
    rowcarbon=k+n+m+k
    rr.extend([rowcarbon]*e); cc.extend(range(e)); vv.extend(emission)
    A=sp.csr_matrix((vv,(rr,cc)),shape=(rowcarbon+1,e))
    supply=1.2*f.sum(axis=1); throughput=1.15*f.sum(axis=0); demand=z.sum(axis=0)
    b=np.r_[np.zeros(k),demand,supply,throughput,emission@witness*2]
    p=Network(A,b,c,u,k+n,m,k,n,emission,witness,{})
    rc=highs(p); re=highs(p,emission)
    ec=float(emission@rc['x']); emin=float(emission@re['x'])
    if ec-emin<1e-8: raise ValueError('No cost-emission tradeoff in this generated case')
    p.b[-1]=emin+beta*(ec-emin)
    p.meta={'seed':seed,'beta':beta,'min_emission':emin,'unconstrained_cost_emission':ec,
            'unconstrained_cost':float(c@rc['x']),'carbon_budget':float(p.b[-1]),
            'factory_positions':posf.tolist(),'warehouse_positions':posw.tolist(),'center_positions':posd.tolist(),
            'generation_note':'two auxiliary LPs calibrate carbon tightness; excluded from timing'}
    return p


def metrics(p,x,y):
    activity=p.A@x; violation=np.r_[np.abs(activity[:p.neq]-p.b[:p.neq]),
                                   np.maximum(activity[p.neq:]-p.b[p.neq:],0)]
    rp=max(float(np.max(violation/(1+np.abs(p.b)))),
           float(np.max(np.maximum(-x,0)/(1+p.u))),float(np.max(np.maximum(x-p.u,0)/(1+p.u))))
    rd=float(np.max(np.maximum(y[p.neq:],0)))/(1+float(np.max(np.abs(y))))
    yf=y.copy(); yf[p.neq:]=np.minimum(yf[p.neq:],0)
    r=p.c-p.A.T@yf; f=float(p.c@x); lower=float(p.b@yf+p.u@np.minimum(r,0))
    gap=abs(f-lower)/(1+abs(f)+abs(lower))
    return dict(objective=f,dual_bound=lower,primal_residual=rp,dual_sign_residual=rd,gap=gap,
                kkt=max(rp,rd,gap),emissions=float(p.emission@x))


def standard(p):
    """Scale rows and append bounded slacks only for upper inequalities."""
    q,e=p.A.shape; ni=q-p.neq; scale=1/np.maximum(1,np.abs(p.b))
    slack=sp.vstack([sp.csr_matrix((p.neq,ni)),sp.eye(ni)],format='csr')
    A=sp.hstack([sp.diags(scale)@p.A,slack],format='csr')
    return A,p.b*scale,np.r_[p.c,np.zeros(ni)],np.r_[p.u,p.b[p.neq:]*scale[p.neq:]],scale


def save(p,path):
    np.savez_compressed(path,data=p.A.data,indices=p.A.indices,indptr=p.A.indptr,shape=p.A.shape,
       b=p.b,c=p.c,u=p.u,neq=p.neq,m=p.m,k=p.k,n=p.n,emission=p.emission,witness=p.witness)


def load(path):
    z=np.load(path)
    A=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=z['shape'])
    return Network(A,z['b'],z['c'],z['u'],int(z['neq']),int(z['m']),int(z['k']),int(z['n']),z['emission'],z['witness'],{})
