"""Educational dense simplex, sparse bound-aware IPM, and modern PDHG variants.

The PDHG restart is a documented residual heuristic, not a reimplementation
of the complete PDLP algorithm. OR-Tools PDLP is a separately labeled baseline.
"""
from time import perf_counter
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor,lu_solve
from model import metrics,standard


def result(x,y,status,k,start,hist,phase1=0,restarts=0,setup=0):
    return dict(x=x,y=y,status=status,iterations=k,seconds=perf_counter()-start,
        history=hist,phase1=phase1,restarts=restarts,setup_seconds=setup)


def record(p,x,y,k,start,hist):
    met=metrics(p,x,y); hist.append(dict(iteration=int(k),seconds=perf_counter()-start,**met)); return met['kkt']


def simplex(p,kind='tableau',tol=1e-6,max_iter=10000,time_limit=30):
    start=perf_counter(); Q,d,cc,upper,scale=standard(p); Q=Q.toarray(); r0,e0=p.A.shape
    r,e=Q.shape; M=np.block([[Q,np.zeros((r,e))],[np.eye(e),np.eye(e)]])
    rhs=np.r_[d,upper]; signs=np.where(rhs<0,-1.,1.); M*=signs[:,None]; rhs*=signs
    q,v=M.shape; cost=np.r_[cc,np.zeros(e)]; F=np.column_stack([M,np.eye(q)])
    basis=np.arange(v,v+q); k=0; hist=[]; phase1=0; status='iteration_limit'
    if kind=='tableau': T=np.column_stack([F,rhs])
    def pivot(row,col):
        T[row]/=T[row,col]; fac=T[:,col].copy(); fac[row]=0
        T[:]-=fac[:,None]*T[row]; basis[row]=col
    for phase in (1,2):
        c=np.r_[np.zeros(v),np.ones(q)] if phase==1 else np.r_[cost,np.zeros(q)]
        allowed=v+q if phase==1 else v
        while True:
            if kind=='tableau':
                xb=T[:,-1]; lam=c[basis]@T[:,v:v+q]; red=c[:allowed]-c[basis]@T[:,:allowed]
            else:
                fac=lu_factor(F[:,basis],check_finite=False); xb=lu_solve(fac,rhs,check_finite=False)
                lam=lu_solve(fac,c[basis],trans=1,check_finite=False); red=c[:allowed]-F[:,:allowed].T@lam
            z=np.zeros(v+q); z[basis]=xb; x=z[:e0]; y=lam[:r0]*signs[:r0]*scale
            if phase==2: residual=record(p,x,y,k,start,hist)
            mask=np.ones(allowed,bool); mask[basis[basis<allowed]]=False
            entering=np.flatnonzero(mask & (red<-1e-10))
            if not len(entering): break
            if k>=max_iter or perf_counter()-start>time_limit:
                return result(x,y,'time_limit' if perf_counter()-start>time_limit else status,k,start,hist,phase1)
            col=int(entering[0]); direction=T[:,col] if kind=='tableau' else lu_solve(fac,F[:,col],check_finite=False)
            pos=np.flatnonzero(direction>1e-10)
            if not len(pos): raise ArithmeticError('No valid leaving pivot')
            ratios=np.maximum(xb[pos],0)/direction[pos]; small=ratios.min()
            tied=pos[np.abs(ratios-small)<=1e-10*(1+abs(small))]; row=int(tied[np.argmin(basis[tied])])
            if kind=='tableau': pivot(row,col)
            else: basis[row]=col
            k+=1
        if phase==1:
            if c@z>1e-7*(1+max(rhs)): return result(x,y,'infeasible',k,start,hist,k)
            for row in np.flatnonzero(basis>=v):
                if kind=='tableau': transformed=T[row,:v]
                else: transformed=lu_solve(lu_factor(F[:,basis]),M)[row]
                candidates=[j for j in np.flatnonzero(np.abs(transformed)>1e-10) if j not in basis]
                if not candidates: raise ArithmeticError('Dependent standard-form row')
                if kind=='tableau': pivot(row,candidates[0])
                else: basis[row]=candidates[0]
                k+=1
            phase1=k
    return result(x,y,'converged' if residual<=tol else 'numerical_failure',k,start,hist,phase1)


def _step(v,d,f=.995):
    mask=d<0
    return min(1.,f*float(np.min(-v[mask]/d[mask]))) if np.any(mask) else 1.


def ipm(p,tol=1e-6,max_iter=100,time_limit=30):
    start=perf_counter(); A,b,c,u,scale=standard(p); AT=A.T.tocsr(); e0=len(p.c)
    e=len(c); x=.5*u; y=np.zeros(len(b)); s=np.ones(e); t=np.ones(e); hist=[]
    status='iteration_limit'
    for k in range(max_iter+1):
        if record(p,x[:e0],y*scale,k,start,hist)<=tol: status='converged'; break
        if k==max_iter: break
        if perf_counter()-start>time_limit: status='time_limit'; break
        v=u-x; rp=A@x-b; rd=c-AT@y-s+t; mu=(x@s+v@t)/(2*e)
        D=1/(s/x+t/v); fac=splu((A.multiply(D)@AT).tocsc())
        def direction(rcx,rcv):
            h=-rd-rcx/x+rcv/v; dy=fac.solve(-rp-A@(D*h)); dx=D*(AT@dy+h)
            return dx,dy,(-rcx-s*dx)/x,(-rcv+t*dx)/v
        dx,dy,ds,dt=direction(x*s,v*t)
        ap=min(_step(x,dx,1),_step(v,-dx,1)); ad=min(_step(s,ds,1),_step(t,dt,1))
        mua=((x+ap*dx)@(s+ad*ds)+(v-ap*dx)@(t+ad*dt))/(2*e); sigma=np.clip((mua/mu)**3,0,1)
        dx,dy,ds,dt=direction(x*s+dx*ds-sigma*mu,v*t-dx*dt-sigma*mu)
        ap=min(_step(x,dx),_step(v,-dx)); ad=min(_step(s,ds),_step(t,dt))
        x+=ap*dx; y+=ad*dy; s+=ad*ds; t+=ad*dt
    return result(x[:e0],y*scale,status,k,start,hist)


def pdhg(p,tol=1e-6,max_iter=30000,time_limit=20,precondition=True,restart=True,row_scale=False):
    start=perf_counter(); scale=1/np.maximum(1,np.abs(p.b)) if row_scale else np.ones(len(p.b)); A=(sp.diags(scale)@p.A).tocsr(); AT=A.T.tocsr(); b=p.b*scale
    absolute=abs(A); colsum=np.asarray(absolute.sum(axis=0)).ravel(); rowsum=np.asarray(absolute.sum(axis=1)).ravel()
    if precondition: tau=.95/colsum; sigma=.95/rowsum
    else:
        # Safe operator-norm upper bound, no iterative norm estimator.
        step=.95/np.sqrt(colsum.max()*rowsum.max()); tau=np.full(len(p.c),step); sigma=np.full(len(b),step)
    x=np.zeros(len(p.c)); xb=x.copy(); y=np.zeros(len(b)); hist=[]; count=0; restarts=0
    sx=np.zeros_like(x); sy=np.zeros_like(y); epoch_R=record(p,x,y,0,start,hist); status='iteration_limit'
    for k in range(1,max_iter+1):
        y+=sigma*(b-A@xb); y[p.neq:]=np.minimum(y[p.neq:],0)
        old=x; x=np.clip(x-tau*(p.c-AT@y),0,p.u); xb=2*x-old
        sx+=x; sy+=y; count+=1
        if k%50==0 or k==max_iter:
            res=record(p,x,y*scale,k,start,hist)
            if res<=tol: status='converged'; break
            if restart:
                ax=sx/count; ay=sy/count; av=metrics(p,ax,ay*scale)['kkt']
                if count>=100 and (min(res,av)<=.25*epoch_R or count>=1000):
                    if av<res: x=ax.copy(); y=ay.copy(); res=av
                    xb=x.copy(); sx.fill(0); sy.fill(0); count=0; epoch_R=res; restarts+=1
            if perf_counter()-start>time_limit: status='time_limit'; break
    # A restart may have replaced the last checkpoint by an averaged iterate.
    record(p,x,y*scale,k,start,hist)
    if hist[-1]['kkt']<=tol: status='converged'
    return result(x,y*scale,status,k,start,hist,restarts=restarts)


def pdlp(p,tol=1e-6,time_limit=20):
    from ortools.pdlp.python import pdlp as native
    from ortools.pdlp import solvers_pb2
    start=perf_counter(); qp=native.QuadraticProgram()
    qp.objective_vector=p.c; qp.variable_lower_bounds=np.zeros_like(p.u); qp.variable_upper_bounds=p.u
    lo=p.b.copy(); lo[p.neq:]=-np.inf
    qp.constraint_lower_bounds=lo; qp.constraint_upper_bounds=p.b; qp.constraint_matrix=p.A.tocsc()
    params=solvers_pb2.PrimalDualHybridGradientParams(); params.num_threads=1
    tc=params.termination_criteria; tc.optimality_norm=solvers_pb2.OPTIMALITY_NORM_L_INF_COMPONENTWISE
    tc.simple_optimality_criteria.eps_optimal_absolute=1e-8
    tc.simple_optimality_criteria.eps_optimal_relative=1e-8; tc.time_sec_limit=time_limit
    setup=perf_counter()-start; native_result=native.primal_dual_hybrid_gradient(qp,params)
    x=np.array(native_result.primal_solution); y=np.array(native_result.dual_solution)
    R=metrics(p,x,y)['kkt']; status='converged' if R<=tol else 'not_certified'
    answer=result(x,y,status,int(native_result.solve_log.iteration_count),start,[],setup=setup)
    answer['native_status']=int(native_result.solve_log.termination_reason)
    return answer


METHODS={'Tableau':lambda p,**kw:simplex(p,'tableau',**kw),
         'Revised':lambda p,**kw:simplex(p,'revised',**kw),'Sparse-IPM':ipm,
         'PR-PDHG':pdhg,'PDLP':pdlp}
