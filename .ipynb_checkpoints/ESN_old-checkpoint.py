""" branched on 10/5 """

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import svd as svd
from itertools import combinations_with_replacement as C_rep
from dataclasses import dataclass, field


FUNCTIONS = {"identity":(lambda x : x), "tanh":torch.nn.Tanh()}

class ESN_mult():
    def __init__(self, N, sigma=0.1, p=1, pin=1, rho=0.9, rseed=0, uC=0, dim=1, idWin=False):
        self.N, self.rho, self.sigma, self.p, self.pin, self.dim = N, rho, sigma, p, pin, dim
        torch.manual_seed(rseed)
        W = torch.rand(N, N) * 2 - 1  # Uniformly distributed between -1 and 1
        W = W * (torch.rand(N, N) < p).float()  # Apply sparsity
     
        if uC != 0:
            U, S, Vh = torch.linalg.svd(W)
            self.W = uC * (U @ torch.eye(N) @ Vh)
        else:
            eigs = torch.linalg.eigvals(W)
            self.W = rho * W / torch.max(torch.abs(eigs))
        
        torch.manual_seed(rseed + 1)
        if idWin:
            Win = (torch.rand(N) * 2 - 1) * sigma
            Win = Win * (torch.rand(N) < pin).float()
            self.Win = Win.repeat(dim, 1).T
        else:
            self.Win = (torch.rand(N, dim) * 2 - 1) * sigma
            self.Win = self.Win * (torch.rand(N, dim) < pin).float()

    def run_washout(self, u, Two, actf=None, f=torch.tanh, X0=None): 
        if actf is not None:
            f = FUNCTIONS[actf]
        if X0 is None:
            X0 = torch.ones(self.N)
        
        if u.shape[0] != self.dim:
            print(f"Input dimension error: input must be shape ({self.dim}, T)")
        
        T = u.shape[1]
        X = torch.zeros((T, self.N))
        X[0] = X0
        for t in range(1, T):
            X[t] = f(self.W @ X[t - 1] + self.Win @ u[:, t - 1])
        
        Xwo = torch.hstack([X[Two:], torch.ones(T - Two, 1)])
        return Xwo

    
@dataclass
class Target_Info:
    tar_f: torch.tensor
    maxdelay: list[int]
    degree: list[int]
    maxddset: list[list[int]]
    
    
def MCwithPI_general(u, Xwo, maxtau):       
    try:
        dim = u.shape[0]
    except IndexError:
        dim = 1
        u = u.unsqueeze(0)  # Convert 1D input to 2D
    
    T = Xwo.shape[0]
    Two = u.shape[1] - T
    piX = torch.linalg.pinv(Xwo)
    
    taus = torch.arange(1, maxtau)
    mfs = torch.tensor(())
    for d in range(dim):
        y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
        normalizer = torch.diag(y_matrix @ y_matrix.T)
        mf = torch.diag((y_matrix @ Xwo) @ (piX @ y_matrix.T)) / normalizer
        mfs = torch.cat((mfs,mf.unsqueeze(0)),0)
    return mfs


def MC_cSVD_old(u, Xwo, maxtau, sur_sets=1, ret_sur=False,raw_mfs=False, rev_method=True):   
    ## for dim ... ##  version of cSVD
    try:
        dim = u.shape[0]
    except IndexError:
        dim = 1
        u = u.unsqueeze(0)  # Convert 1D input to 2D
    N = Xwo.shape[1]
    T = Xwo.shape[0]
    Two = u.shape[1] - T
    U,sigma,_ = torch.linalg.svd(Xwo,full_matrices=False)
    if sigma[N-1] != 0: rank = N
    else : rank = torch.where(sigma==0)[0][0]
    P = U[:,:rank]
    # P: T * rank
    
    taus = torch.arange(1, maxtau)
    mfs = torch.tensor(())
    for d in range(dim):
        ## calculate surrogate value
        agg_sur = 0
        for i in range(sur_sets):            
            y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
            shfld_y = y_matrix[:,torch.randperm(y_matrix.shape[1])]
            norms = torch.norm(shfld_y, dim=1).unsqueeze(0).T
            sur = torch.sum(((shfld_y/norms) @ P)**2, dim=1)     
            agg_sur = agg_sur + torch.sum(sur)/sur.shape[0]
        sur_value = agg_sur/sur_sets        
        
        # y : tau * T
        y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
        norms = torch.norm(y_matrix, dim=1).unsqueeze(0).T
        # use normalized input
        mf = torch.sum(((y_matrix/norms) @ P)**2, dim=1)        
        #### apply surrogate
        
        if rev_method : 
            mf = (mf-sur_value)/(1-sur_value)
        else :
            # substract surrogate value according to mf
            mf = mf - (1-mf)*sur_value
            mf = mf*(mf>0)
        
        mfs = torch.cat((mfs,mf.unsqueeze(0)),0)
    if ret_sur :
        return mfs, sur_value
    else : return mfs


def MC_cSVD(u, Xwo, maxtau, sur_sets=1, ret_sur=False,raw_mfs=False, rev_method=True,ret_all=False):   
    try:
        dim = u.shape[0]
    except IndexError:
        dim = 1
        u = u.unsqueeze(0)  # Convert 1D input to 2D
    N = Xwo.shape[1]
    T = Xwo.shape[0]
    Two = u.shape[1] - T
    U,sigma,_ = torch.linalg.svd(Xwo,full_matrices=False)
    if sigma[N-1] != 0: rank = N
    else : rank = torch.where(sigma==0)[0][0]
    P = U[:,:rank]
    # P: T * rank
    taus = torch.arange(1, maxtau)        
    # y_matrix : dim * tau * T
    y_matrix = torch.tensor(())
    for d in range(dim):
        y_matrix = torch.cat((y_matrix,torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus]).unsqueeze(0)),0)
    norms = torch.norm(y_matrix, dim=2).unsqueeze(2)
    
    # mfs : d * tau
    mfs = torch.sum(((y_matrix / norms) @ P)**2,dim=2) 
    raw_res = mfs
    
    ## calculate surrogate value
    agg_sur = torch.zeros(dim)
    for i in range(sur_sets):            
        shfld_y = y_matrix[:,:,torch.randperm(y_matrix.shape[2])]
        sur = torch.sum(((shfld_y / norms) @ P)**2,dim=(1,2))/(maxtau-1)
        agg_sur = agg_sur + sur
    sur_value = agg_sur/sur_sets

    mfs_rev = (raw_res - sur_value.unsqueeze(1))/(1-sur_value.unsqueeze(1))
    mfs_lin = mfs - (1-mfs)*sur_value.unsqueeze(1)
    if rev_method :  mfs = mfs_rev
    else : mfs = mfs_lin
    #mfs = mfs*(mfs>0)
    
    if ret_sur :  return mfs, sur_value
    elif raw_mfs: return mfs,raw_res,sur_value
    elif ret_all: return raw_res, mfs_lin, mfs_rev, sur_value
    else : return mfs
    



def MCwithPI_general_with_surrogate(u, Xwo, maxtau,sur_samples = 0, sur_scalar = 1.8):       
    try:
        dim = u.shape[0]
    except IndexError:
        dim = 1
        u = u.unsqueeze(0)  # Convert 1D input to 2D
        
    T = Xwo.shape[0]
    Two = u.shape[1] - T
    piX = torch.linalg.pinv(Xwo)
    
    taus = torch.arange(1, maxtau)
    mfs = torch.tensor(())

    if sur_samples != 0 :
        for d in range(dim):
            y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
            normalizer = torch.diag(y_matrix @ y_matrix.T)
            mf = torch.diag((y_matrix @ Xwo) @ (piX @ y_matrix.T)) / normalizer
            
            # if mf is less than 1/5 of mf[0], use surrogate to truncate irrelevantly small values
            tau_st = torch.argwhere(mf<mf[0]/5)[0][0]
            sur_taus = torch.arange(tau_st,maxtau)
            agg_sur = torch.zeros(maxtau-tau_st)
            for i in range(sur_samples):            
                y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in sur_taus])
                shfld_y = y_matrix[:,torch.randperm(y_matrix.shape[1])]
                sur_matrix = (shfld_y @ Xwo @ piX @ shfld_y.T)
                normalizer = torch.diag(shfld_y @ shfld_y.T)
                sur = torch.diag(sur_matrix) / normalizer
                agg_sur = agg_sur + sur
            thr_sur = agg_sur/sur_samples * sur_scalar
            mf[tau_st-1:maxtau-1] = mf[tau_st-1:maxtau-1]*(mf[tau_st-1:maxtau-1]>thr_sur)
            mfs = torch.cat((mfs,mf.unsqueeze(0)),0)
            
        return mfs
    
    else:
        for d in range(dim):
            y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
            mf_matrix = (y_matrix @ Xwo @ piX @ y_matrix.T)
            normalizer = torch.diag(y_matrix @ y_matrix.T)
            mf = torch.diag(mf_matrix) / normalizer
            mfs = torch.cat((mfs,mf.unsqueeze(0)),0)
        return mfs
   

def MCwithPI_general_newsur(u, Xwo, maxtau,sur_sets):       
    try:
        dim = u.shape[0]
    except IndexError:
        dim = 1
        u = u.unsqueeze(0)  # Convert 1D input to 2D
        
    T = Xwo.shape[0]
    Two = u.shape[1] - T
    piX = torch.linalg.pinv(Xwo)
    
    taus = torch.arange(1, maxtau)
    mfs = torch.tensor(())
        
    for d in range(dim):
        agg_sur = 0
        ## calculate surrogate value
        for i in range(sur_sets):            
            y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
            shfld_y = y_matrix[:,torch.randperm(y_matrix.shape[1])]
            normalizer = torch.diag(shfld_y @ shfld_y.T)
            sur = torch.diag((shfld_y @ Xwo) @ (piX @ shfld_y.T)) / normalizer
            agg_sur = agg_sur + torch.sum(sur)/sur.shape[0]
        sur_value = agg_sur/sur_sets
        # standard memory function calculation
        y_matrix = torch.stack([u[d][Two - tau:Two + T - tau] for tau in taus])
        normalizer = torch.diag(y_matrix @ y_matrix.T)
        mf = torch.diag((y_matrix @ Xwo) @ (piX @ y_matrix.T)) / normalizer
        #### apply surrogate
        # substract surrogate value according to mf
        mf = mf - (1-mf)*sur_value
        mfs = torch.cat((mfs,mf.unsqueeze(0)),0)
        
    return mfs
    
    
    
def calc_capacity(Xwo,targets,sur_sets=10,ret_all = False,forced_sur=None):
    if Xwo.shape[0] == targets.shape[1]:
        T = Xwo.shape[0]
    N = Xwo.shape[1]
    units = targets.shape[0]
    U,sigma,_ = torch.linalg.svd(Xwo,full_matrices=False)
    if sigma[N-1] != 0: rank = N
    else : rank = torch.where(sigma==0)[0][0]
    P = U[:,:rank]
    # P: T * rank
    # targets : (target units) * Ttrain 
    norms = torch.norm(targets, dim=1).unsqueeze(1)
    # capacities : target units
    capacities = torch.sum(((targets / norms) @ P)**2,dim=1) 
    raw_res = capacities
    
    if forced_sur==None:
        ## calculate surrogate value
        agg_sur = torch.zeros(units)
        for i in range(sur_sets):            
            shfld_tar = targets[:,torch.randperm(targets.shape[1])]
            sur = torch.sum(((shfld_tar / norms) @ P)**2,dim=1)
            agg_sur = agg_sur + sur
        sur_value = agg_sur/sur_sets
    else:sur_value = forced_sur
    # apply surrogate
    c_rev = (raw_res - sur_value)/(1-sur_value)
    c_lin = raw_res - (1-raw_res)*sur_value
    #mfs = mfs*(mfs>0)
    if ret_all: return raw_res, c_lin, c_rev, sur_value
    else : return c_rev

def polynomials(base):
    if base=="legendre":
        return legendre

def legendre(u,degree):
    if degree == 0 : return 1
    elif degree == 1 : return u
    elif degree == 2 : return (3*u**2-1)/2
    elif degree == 3 : return (5*u**3-3*u)/2
    elif degree == 4 : return (35*u**4 - 30*u**2 +3)/8
    elif degree == 5 : return (63*u**5 -70*u**3 + 15*u)/8
    elif degree > 5: print(f"{degree}th order not implemented")


def make_targets_old(u,maxddsets,Two,poly="legendre"):
    dims = u.shape[0]
    T = u.shape[1] - Two
    f = polynomials(poly)
    ## exmaple of dd_inds : [[0,0,0],[0,0,1],...]
    ## ->corresponds to ddset of [3,0,...],[2,1,...]
    ## shape of dd_inds: untis * dgr
    targets = torch.tensor(())
    dgrs = torch.tensor((),dtype = torch.int)
    for maxdd in maxddsets:
        ## dd =  [dgr,dly]
        maxdgr = maxdd[0]
        maxdly = maxdd[1]
        if maxdd[0]==1:
            for dim in dims:
                tar = u[dim][Two - maxdly:Two + T - maxdly]
                targets = torch.cat((targets,tar.unsqueeze(0)),0)
        
        ## set of delays representing non zero degrees
        ## ex) [1,1]   -> (3* u(t-1)**2 -1)/2
        ##     [1,2,3] -> u(t-1)*u(t-2)*u(t-3)
        dd_inds = list(C_rep(range(dims * maxdly),maxdgr))
        
        ## make targets
        ## targets : units * Ttrain
        for unit,inds in enumerate(dd_inds) :
            tar = torch.ones(T)
            scanned=[]
            for i in range(maxdgr):
                if inds[i] in scanned : continue
                dgr = inds.count(inds[i])
                scanned.append(inds[i])
                dim = int(inds[i]/maxdly)
                dly = 1+ inds[i]%maxdly
                tar *= f(u[dim][Two-dly:Two-dly+T],degree=dgr)
            targets = torch.cat((targets,tar.unsqueeze(0)),0)
        print(f"{maxdgr} degree:{len(dd_inds)} output units")
        dgrs = torch.cat((dgrs,torch.ones(len(dd_inds))*maxdgr),0)
    
    return targets,dgrs           




def make_targets(u,maxddsets,Two,poly="legendre"):
    dims = u.shape[0]
    T = u.shape[1] - Two
    f = polynomials(poly)

    ## shape of dd_delays: untis * dgr
    ## dd_delays: set of delays representing non zero degrees
    ## exmaple of dd_delays : [[0,0,0],[0,0,1],...]
    ## ->corresponds to ddset of [3,0,...],[2,1,0,...]
    
    # IF input dim:2, max delay:5 
    # [4,5,7]->[[0,0,0,1,1],
    #           [0,1,0,0,0]]  
    
    ##  with legendre function 
    ##  [1,1]   -> (3* u(t-1)**2 -1)/2
    ##  [1,2,3] -> u(t-1)*u(t-2)*u(t-3)
        
    """init return values"""
    targets = torch.tensor(())
    dgrs = []
    maxdelays = []
    
    for maxdd in maxddsets:
        ## dd =  [dgr,dly]
        maxdgr = maxdd[0]
        maxdly = maxdd[1]
        if maxdd[0]==1:
            for dim in dims:
                tar = u[dim][Two - maxdly:Two + T - maxdly]
                targets = torch.cat((targets,tar.unsqueeze(0)),0)
        dd_delays = list(C_rep(range(dims * (maxdly)),maxdgr))
        ## make targets
        ## targets : units * Ttrain
        for unit,delays in enumerate(dd_delays) :
            tar = torch.ones(T)
            scanned=[]
            maxdelay=0
            tau_ipc=0
            for i in range(maxdgr):
                if delays[i] in scanned : continue
                dgr = delays.count(delays[i])
                scanned.append(delays[i])
                dim = int(delays[i]/maxdly)
                dly = 1+ delays[i]%maxdly
                tau_ipc=max(tau_ipc,dly)
                tar *= f(u[dim][Two-dly:Two-dly+T],degree=dgr)
            maxdelays.append(tau_ipc)
            targets = torch.cat((targets,tar.unsqueeze(0)),0)
            
        print(f"{maxdgr} degree:{len(dd_delays)} target functions")
        dgrs += ([maxdgr]*len(dd_delays)) 
        
    target_info = Target_Info(targets,maxdelays,dgrs,maxddsets)
    return target_info   
            

def ipc_tau(ipcs,degrees,maxdelays,maxddsets):
    ipc_tau=[]
    offset = maxddsets[0][0]
    
    maxdegree = len(maxddsets)
    ipc_reshape = [None]*maxdegree
    ipc_tau = [None]*maxdegree
    for deg,delay in maxddsets:
        ipc_reshape[deg-offset] = [torch.tensor(())]*(delay)
        ipc_tau[deg-offset] = torch.empty(delay)
        
    for i,deg in enumerate(degrees):
        ipc_reshape[deg-offset][maxdelays[i]-1]=torch.cat((ipc_reshape[deg-offset][maxdelays[i]-1],ipcs[i].unsqueeze(0))) 
        
    for deg,maxdelay in maxddsets:
        for delay in range(maxdelay):
            ipc_tau[deg-offset][delay] = torch.sum(ipc_reshape[deg-offset][delay])   
    
    ## list of tensors
    ## maxdegree length list
    ## maxdelay length tensor
    return ipc_tau
    
    
    
 
"""
def calc_ipc(target_inf,Xwo,sur_sets):

    tar_func = target_info.tar_f
    dgrs = torch.tensor(target_function.degree)
    dlyed_u = tar_func[torch.argwhere(dgrs==1)]
    -,-,mfs,sur = ESN.calc_capacity(Xwo,dlyed_u,sur_sets=sur_sets)
    tar_ipc = tar_func[torch.argwhere(dgrs==1)]
    capacities = ESN.calc_capacity(Xwo,tar_ipc,forced_sur=sur)
    ipc = torch.cat((mfs,capacities))
    return ipc




"""