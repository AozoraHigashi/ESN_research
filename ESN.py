import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.linalg import svd as svd
from itertools import combinations_with_replacement as C_rep
from dataclasses import dataclass, field
import time

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
    def get_W(self):return self.W
    
@dataclass
class Target_Info:
    tar_f: torch.tensor
    delay: list[int]
    degree: list[int]
    in_dim : list[list[int]]    #starts from 0
    maxddset: list[list[int]]
    
    def slice(self,stt:int,end:int):
        return Target_Info(self.tar_f[stt:end],self.maxdelay[stt:end],self.degree[stt:end],self.in_dim[stt:end],self.maxddset)
    

def cat_tar(tar1:Target_Info,tar2:Target_Info):
        tar_ap = torch.cat((tar1.tar_f,tar2.tar_f))
        
        dl_ap = tar1.delay+tar2.delay
        
        dg_ap = tar1.degree+tar2.degree
        id_ap = tar1.in_dim+tar2.in_dim     
        dd_ap = tar1.maxddset+tar2.maxddset
        return Target_Info(tar_ap,dl_ap,dg_ap,id_ap,dd_ap)


@dataclass
class IPC:
    val: torch.tensor
    delay: list[list[int]]
    degree: list[int]
    in_dim : list[list[int]]    #starts from 0
    maxddset: list[list[int]]

    def slice(self,stt:int,end:int):
        return IPC(self.val[stt:end],self.maxdelay[stt:end],self.degree[stt:end],self.in_dim[stt:end],self.maxddset)      

    # Method to get values for a specific degree
    def get_val_by_degree(self, target_degree: int) -> torch.tensor:
        indices = [i for i, deg in enumerate(self.degree) if deg == target_degree]
        return self.val[indices]
    
    # Method to get total capacity for a specific degree
    def ipc_by_degree(self, target_degree: int) -> torch.tensor:
        return torch.sum(self.get_val_by_degree(target_degree))
    
    # Method to search ipc data by index
    def get_by_ind(self, index: int):
        return IPC(self.val[index],self.maxdelay[index],self.degree[index],self.in_dim[index],self.maxddset)      

def IPC_w_targetinfo(capacities:torch.tensor ,target_info:Target_Info):
    if capacities.shape[0]!=target_info.tar_f.shape[0]:
        print(f"input length does not match: {capacities.shape[0]} ,{target_info.tar_f.shape[0]}",)
    return IPC(capacities,target_info.delay,target_info.degree,target_info.in_dim,target_info.maxddset)
    
def cat_ipc(ipc1:IPC,ipc2:IPC):
        tar_ap = torch.cat((ipc1.tar_f,ipc2.tar_f))
        
        dl_ap = ipc1.maxdelay+ipc2.maxdelay

        dg_ap = ipc1.degree+ipc2.degree
        id_ap = ipc1.in_dim+ipc2.in_dim
        dd_ap = ipc1.maxddset+ipc2.maxddset
        return IPC(tar_ap,dl_ap,dg_ap,id_ap,dd_ap)



def MC_cSVD(u, Xwo, maxtau, sur_sets=20, ret_all=False):   
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
    if ret_all: return raw_res, mfs_lin, mfs_rev, sur_value
    else : return mfs

def MC_cSVD_theoretical(u, Xwo, maxtau, sur_sets=20, ret_all=False):   
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
    mfs_theor = (raw_res - sur_value.unsqueeze(1)) / (1-(sur_value.unsqueeze(1)/(N-1)))

    if ret_all: return raw_res, mfs_theor, mfs_rev, sur_value
    else : return mfs

def MC_cSVD_asym(u, Xwo, maxtau, sur_sets=1, ret_all=False):   
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
    means = torch.mean(y_matrix, dim=2).unsqueeze(2)
    norms = torch.norm(y_matrix-means, dim=2).unsqueeze(2)
    normal_y = ((y_matrix-means) / norms)
    print(torch.norm(normal_y,2))
    # mfs : d * tau
    mfs = torch.sum((normal_y @ P)**2,dim=2) 
    raw_res = mfs
    ## calculate surrogate value
    agg_sur = torch.zeros(dim)
    for i in range(sur_sets):            
        torch.manual_seed(torch.seed())
        shfld_y = normal_y[:,:,torch.randperm(y_matrix.shape[2])]
        sur = torch.sum((shfld_y @ P)**2,dim=(1,2))/(maxtau-1)
        agg_sur = agg_sur + sur
    sur_value = agg_sur/sur_sets


    mfs_rev = (raw_res - sur_value.unsqueeze(1))/(1-sur_value.unsqueeze(1))
    mfs_lin = mfs - (1-mfs)*sur_value.unsqueeze(1)
    if ret_all: return raw_res, mfs_lin, mfs_rev, sur_value
    else : return mfs
    
 
    
def calc_capacity_module(Xwo,targets,sur_sets=10,ret_all = False,forced_sur=None,thr_scale=None,mean_normalization=True):
    if Xwo.shape[0] == targets.shape[1]:
        T = Xwo.shape[0]
    N = Xwo.shape[1]
    units = targets.shape[0]
    if mean_normalization:
        Xmean = torch.mean(Xwo,0)
        Xwo = Xwo-Xmean
    U,sigma,_ = torch.linalg.svd(Xwo,full_matrices=False)
    if sigma[N-1] != 0: rank = N
    else : rank = torch.where(sigma==0)[0][0]
    P = U[:,:rank]
    # P: T * rank
    # targets : (number of bases) * Ttrain 
    
    ## implemented input mean normalzation
    means = torch.mean(targets, dim=1).unsqueeze(1)
    norms = torch.norm(targets-means, dim=1).unsqueeze(1)
    if mean_normalization : target_hat = (targets-means)/norms
    else: target_hat = targets/norms
    # capacities : number of bases
    #capacities = torch.sum(((targets / norms) @ P)**2,dim=1) 
    capacities = torch.sum((target_hat @ P)**2,dim=1) 
    raw_res = capacities
    
    if forced_sur==None:
        ## calculate surrogate value
        agg_sur = torch.zeros(units)
        max_sur=0
        for i in range(sur_sets):            
            shfld_tar = target_hat[:,torch.randperm(targets.shape[1])]
            sur = torch.sum(((shfld_tar) @ P)**2,dim=1)
            agg_sur = agg_sur + sur
            if i==0 : max_sur= max(sur)
        sur_value = agg_sur/sur_sets
    else:
        sur_value = forced_sur
        max_sur = forced_sur
    # apply surrogate
    c_rev = (raw_res - sur_value)/(1-sur_value)
#    c_lin = raw_res - (1-raw_res)*sur_value
    if thr_scale == None : 
        if ret_all:print("set threshold scale value: thr_scale=",thr_scale)
        c_thr=None
    else : 
        c_thr=raw_res*((raw_res - max_sur*thr_scale)>0)
        # threshold and scaling combined
        c_thr_scale = (raw_res - sur_value)/(1-sur_value)*((raw_res - max_sur*thr_scale)>0)

    #mfs = mfs*(mfs>0)
    if ret_all: return raw_res, c_thr, c_thr_scale, c_rev, sur_value
    else : return c_rev



## avoid OOM error
def calc_capacity(Xwo,targets,sur_sets=10,ret_all = False,forced_sur=None,thr_scale=None,mean_normalization=False):
    try : res = calc_capacity_module(Xwo,targets,sur_sets,ret_all,forced_sur,thr_scale,mean_normalization) 
    except RuntimeError: 
        tar1 = targets[:int(targets.shape[0]/2)]
        tar2 = targets[int(targets.shape[0]/2):]
        res1 = calc_capacity(Xwo,tar1,sur_sets,ret_all,forced_sur,thr_scale)
        res2 = calc_capacity(Xwo,tar2,sur_sets,ret_all,forced_sur,thr_scale)
        res = torch.cat((res1,res2))
    return res




def calc_capacity_asym(Xwo,targets,sur_sets=10,ret_all = False,forced_sur=None):
    if Xwo.shape[0] == targets.shape[1]:
        T = Xwo.shape[0]
    N = Xwo.shape[1]
    units = targets.shape[0]
    U,sigma,_ = torch.linalg.svd(Xwo,full_matrices=False)
    if sigma[N-1] != 0: rank = N
    else : rank = torch.where(sigma==0)[0][0]
    P = U[:,:rank]
    # P: T * rank
    # targets : (number of bases) * Ttrain 
    means = torch.mean(targets, dim=1).unsqueeze(1)
    norms = torch.norm(targets-means, dim=1).unsqueeze(1)
    target_hat = (targets-means)/norms
    # capacities : number of bases
    #capacities = torch.sum(((targets / norms) @ P)**2,dim=1) 
    capacities = torch.sum((target_hat @ P)**2,dim=1) 
    raw_res = capacities
    
    if forced_sur==None:
        ## calculate surrogate value
        agg_sur = torch.zeros(units)
        for i in range(sur_sets):            
            shfld_tar = target_hat[:,torch.randperm(targets.shape[1])]
            sur = torch.sum((shfld_tar @ P)**2,dim=1)
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
        return torch._C._special.special_legendre_polynomial_p

    
""" 
# naive implmentation
def legendre(u,degree):
    if degree == 0 : return 1
    elif degree == 1 : return u
    elif degree == 2 : return (3*u**2-1)/2
    elif degree == 3 : return (5*u**3-3*u)/2
    elif degree == 4 : return (35*u**4 - 30*u**2 +3)/8
    elif degree == 5 : return (63*u**5 -70*u**3 + 15*u)/8
    elif degree > 5: print(f"{degree}th order not implemented")
"""

    

def make_targets(u,maxddsets,Two,poly="legendre"):
    dims = u.shape[0]
    T = u.shape[1] - Two
    f_poly = polynomials(poly)
    maxddsets = np.array(maxddsets)
    
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
    targets = torch.zeros((1,T))

    dgrs = []
#    maxdelays = []
    in_dim =[]
    delay_s =[]
    setdegrees = maxddsets[:,0]
    setdelays = maxddsets[:,1]
    dly_range = [max(setdelays[np.where(setdegrees>=dgr)[0][0]:]) for dgr in np.arange(1,max(setdegrees)+1)]
    ## table of univariate bases, used as lookup table for fast computation of target functions
    ## shape: (maxdegree,dims,T+Two)
    ## initialize table with raw values at basis_table[0]
    st=time.time()
    basis_table=u.unsqueeze(0)
    for dgr in np.arange(2,max(setdegrees)+1):
        basis_table= torch.cat((basis_table,f_poly(u,dgr).unsqueeze(0)),0)
    print(r"basis table creation:%.3f s"%(time.time()-st))    

    for i,maxdd in enumerate(maxddsets):
        ## dd =  [dgr,dly]
        maxdgr = maxdd[0]
        maxdly = maxdd[1]

        if maxdgr==1:
            for dim in range(dims):
                for tau in np.arange(1,maxdly+1):
                    targets = torch.cat((targets,u[dim][Two-tau:Two-tau+T].unsqueeze(0)),0)
                delay_s+= [[n]for n in np.arange(1,maxdly+1)]
                dgrs += ([1]*maxdly)
                in_dim += ([dim]*maxdly)            
            print(f"1 degree:{maxdly*dims} target functions")
            continue
        set_of_delays = list(C_rep(range(dims * (maxdly)),maxdgr))
        ## make targets
        ## targets : units * Ttrain
        for delays in set_of_delays :
            tar = torch.ones(T)
            scanned=[]
            dim_in =[]
            delay_set =[]
#            tau_ipc=0
            for i in range(maxdgr):
                if delays[i] in scanned : continue
                dgr = delays.count(delays[i])
                scanned.append(delays[i])
                dly = 1+ delays[i]%maxdly
#                tau_ipc=max(tau_ipc,dly)
                dim = int(delays[i]/maxdly)

                delay_set.append(dly)
                tar *= basis_table[dgr-1][dim][Two-dly:Two-dly+T]
                dim_in.append(dim)

            delay_s.append(delay_set)
#            maxdelays.append(tau_ipc)      
            targets = torch.cat((targets,tar.unsqueeze(0)),0)
            """ use cpu 
            targets = torch.cat((targets,tar.unsqueeze(0).cpu()),0)
            """
            in_dim.append(dim_in)
            
        print(f"{maxdgr} degree:{len(set_of_delays)} target functions")
        dgrs += ([maxdgr]*len(set_of_delays))
    targets=targets[1:]
    # old version
    # target_info = Target_Info(targets,maxdelays,dgrs  ,maxddsets,in_dim)
    target_info = Target_Info(targets,delay_s,dgrs,in_dim,maxddsets)
    
    return target_info   
            


def ipc_tau_old(ipcs:torch.tensor,degrees:list[list[int]],delays:list[list[int]],in_dims,maxddsets:list[list[int]],mode="sum"):
    ipc_tau=[]
    offset = maxddsets[0][0]
    maxdelays = []
    maxddsets = np.array(maxddsets)
    for delay_set in delays:
        maxdelays.append(max(delay_set))
    
    num_degree = len(maxddsets)

    ipc_reshape = [None]*num_degree
    ipc_tau = [None]*num_degree

    for deg,delay in maxddsets:
        ipc_reshape[deg-offset] = [torch.tensor(())]*(delay)
        ipc_tau[deg-offset] = torch.empty(delay)
        
    for i,deg in enumerate(degrees):
        ipc_reshape[deg-offset][maxdelays[i]-1]=torch.cat((ipc_reshape[deg-offset][maxdelays[i]-1],ipcs[i].unsqueeze(0))) 
        
    for deg,maxdelay in maxddsets:
        for delay in range(maxdelay):
            if mode=="sum" :ipc_tau[deg-offset][delay] = torch.sum(ipc_reshape[deg-offset][delay])
            elif mode=="mean":ipc_tau[deg-offset][delay] = torch.mean(ipc_reshape[deg-offset][delay])   
            else:print("available modes are: sum/mean")
    ## list of tensors
    ## maxdelay length tensor IN num_degree length list
    return ipc_tau

def ipc_tau(ipc:IPC,mode="sum"):
    return ipc_tau_old(ipc.val,ipc.degree,ipc.delay,None,ipc.maxddset,mode=mode)

def ipc_tau_spread(ipc:IPC,mode="sum"):
    degrees=ipc.degree
    delays=ipc.delay
#    in_dims=ipc.in_dim
    maxddsets=ipc.maxddset

    ipc_tau=[]
    offset = maxddsets[0][0]
    num_degree = len(maxddsets)
    ipc_reshape = [None]*num_degree
    ipc_tau = [None]*num_degree
    for deg,delay in maxddsets:
        ipc_reshape[deg-offset] = [torch.tensor(())]*(delay)
        ipc_tau[deg-offset] = torch.empty(delay)
    
    for i,deg in enumerate(degrees):
        # divide ipc by len(delays[i]) to spread the contribution
        num_delays = len(delays[i])
        for dly in delays[i]:
            ipc_reshape[deg-offset][dly-1]=torch.cat((ipc_reshape[deg-offset][dly-1],(ipc.val[i]/num_delays).unsqueeze(0)),0) 
        
    for deg,maxdelay in maxddsets:
        for delay in range(maxdelay):
            if mode=="sum" :ipc_tau[deg-offset][delay] = torch.sum(ipc_reshape[deg-offset][delay])
            elif mode=="mean":ipc_tau[deg-offset][delay] = torch.mean(ipc_reshape[deg-offset][delay])   
            else:print("available modes are: sum/mean")
    ## list of tensors
    ## maxdelay length tensor IN num_degree length list
    return ipc_tau


"""    
def ipc_process(ipc; IPC):
    ipc_tau = ipc_tau(ipc.val,ipc.degree,ipc.maxdelay,ipc.maxddset)
    
    return ipc_tau,
 



def calc_ipc(target_info,Xwo,sur_sets,ret_all=False):

    tar_func = target_info.tar_f
    capacities = ESN.calc_capacity(Xwo,tar_func,forced_sur=None)
    return 


def plot_ipc_tau():



def plot_forgetting_curve():

def plot_by_degree(ipc:IPC, )




"""