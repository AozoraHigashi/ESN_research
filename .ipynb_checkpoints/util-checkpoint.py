import torch
import matplotlib.pyplot as plt
import ESN
import time
import numpy as np

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.double)


def gram_schmidt(vectors,normalize = False):
    """
    Apply Gram-Schmidt orthogonalization to a set of vectors.

    Args:
        vectors (torch.Tensor): A tensor of shape (num_vectors, vector_dim).

    Returns:
        torch.Tensor: Orthogonalized vectors of shape (num_vectors, vector_dim).
    """
    orthogonal_vectors = []
#    orthogonal_vectors = [torch.ones(vectors.size(1))]
    for i in range(vectors.size(0)):
        v = vectors[i]
        for u in orthogonal_vectors:
            v = v - torch.dot(v, u) / torch.dot(u, u) * u
        if normalize:
            v = v/torch.norm(v)
        orthogonal_vectors.append(v)
    return torch.stack(orthogonal_vectors)

def process_tensor(input_tensor, tau, T):
    """
    Process the input tensor to generate orthogonalized delay vectors.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (t + T_tr).
        tau (int): Maximum delay.
        T (int): Length of delay tensor.

    Returns:
        torch.Tensor: Orthogonalized vectors of shape (tau + 1, T).
    """
    # Validate input
    assert input_tensor.dim() == 1, "Input tensor must be 1-dimensional"
    t_Ttr = input_tensor.size(0)
    assert t_Ttr > T, "Input tensor length must be greater than T"
    # Extract tau + 1 vectors from the input tensor
    
    tau_vectors = torch.stack([input_tensor[-T-i:-i] for i in torch.arange(1,tau)])
    tau_vectors = torch.vstack((input_tensor[-T:],tau_vectors))
    # Apply Gram-Schmidt orthogonalization
    orthogonal_vectors = gram_schmidt(tau_vectors)

    return orthogonal_vectors


def ipc_tau_plot(ipc:ESN.IPC,degrees=None,xmax=None):
    ipc_tau_sp_sum = ESN.ipc_tau_spread(ipc=ipc,mode="sum")
    ipc_tau_sp_mean = ESN.ipc_tau_spread(ipc=ipc,mode="mean")
    ipc_tau_sum = ESN.ipc_tau(ipc=ipc,mode="sum")
    ipc_tau_mean = ESN.ipc_tau(ipc=ipc,mode="mean")

    ipc_taus=[[ipc_tau_sp_sum,ipc_tau_sp_mean],
              [ipc_tau_sum,ipc_tau_mean]]
    
    if degrees == None : degrees = range(len(ipc_tau_sum))
    if xmax == None : ipc.maxddset[0,1]
    
    fig,axes = plt.subplots(2,2,figsize=(16,12))
    
    for i,rows in enumerate(axes):
        ipc_row=ipc_taus[i]
        for j,axis in enumerate(rows):
            ipc_tau=ipc_row[j]
            for deg in degrees:
                axis.plot(np.arange(1,ipc_tau[deg].shape[0]+1),ipc_tau[deg].cpu().numpy(),label=f"{deg+1} degree")
##                if xmax == None : 
##     look for tau after which ipc_tau is all zero                    
            axis.set_xticks([1]+list(np.arange(5,xmax,5)))            
            axis.set_xlabel("delay")
            axis.legend()
            axis.grid(True)


    axes[0,0].set_title("spread sum")
    axes[0,1].set_title("spread mean")
    axes[1,0].set_title("maxdelay sum")
    axes[1,1].set_title("maxdelay mean") 

    totIPC = torch.sum(ipc.val).cpu().numpy()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle(r'Ttr = %d, rho=%.2f,$N=%d, IPC=%.1f$'%(Ttrain,rho,N_d,totIPC), fontsize=16)
    fig.suptitle(r'IPC values - delay timestep,  IPC = '+str(totIPC), fontsize=16)
    fig.show()

def print_ipc(ipc:ESN.IPC):
    degree_num=len(ipc.maxddset)
    cap_deg=np.array(())
    for deg in range(degree_num):
        tempcap = torch.sum(ipc.get_val_by_degree(deg+1)).cpu().numpy()
        cap_deg = np.append(cap_deg,tempcap)
        print(f"{deg+1} deg ipc:",tempcap)
    print("total ipc:",float(np.sum(cap_deg)))

    
def OU_random_sampling(rseed,theta=1, mu=0, sigma=0.5, T=100000, disc_step=1000,Y_INIT=0):
    if T < disc_step:
        print(f"T must be larger than disc_step: {T} < {disc_step}")
    
    T_INIT = 0
    T_END = T // disc_step
    DT = 1.0 / disc_step
    N = float(T_END - T_INIT) / DT
    TS = torch.arange(T_INIT, T_END + DT, DT)
    assert TS.size(0) == int(N) + 1

    Y_INIT = 0
    ys = torch.zeros(TS.size(0))
    delta_y = torch.zeros(TS.size(0))
    ys[0] = Y_INIT

    torch.manual_seed(rseed)
    noise = torch.normal(0,1,(T,),device="cuda:0")
    eta = noise * torch.sqrt(torch.tensor(DT))  
    
    for i in range(TS.size(0)):
        delta_y[i] = -theta * (ys[i - 1] - mu) * DT + sigma * eta[i-1]
        ys[i] = ys[i - 1] + delta_y[i-1]

    """     bugged version ?   
    delta_y = torch.tensor([-theta * (ys[i - 1] - mu) * DT + sigma * eta[i-1] for i in range(1, TS.size(0))])
    ys = torch.tensor([ys[i - 1] + delta_y[i-1] for i in range(1, TS.size(0))])
    """   
    return TS[1:], ys[1:], delta_y[1:]
    

def Ornstein_Uhlenbeck(noise:torch.tensor,theta=1,mu=0,sigma=0.5,disc_step=1000,Y_INIT=0):
    ## !! noise have to be standard gaussian 
    ## T have to be divedable by disc_step, probably    
    T = noise.size(0)
    if T == 1: 
        T = noise.size(1)
        noise = noise[0]
        
    if T < disc_step:
        print(f"noise length must be larger than disc_step: {T} < {disc_step}")
    
    T_INIT = 0
    T_END = T // disc_step
    DT = 1.0 / disc_step
    N = float(T_END - T_INIT) / DT
    TS = torch.arange(T_INIT, T_END + DT, DT)
    assert TS.size(0) == int(N) + 1
    
    ys = torch.zeros(TS.size(0))
    delta_y = torch.zeros(TS.size(0))
    ys[0] = Y_INIT
    
    eta = noise * torch.sqrt(torch.tensor(DT))  
    for i in range(TS.size(0)):
        delta_y[i] = -theta * (ys[i - 1] - mu) * DT + sigma * eta[i-1]
        ys[i] = ys[i - 1] + delta_y[i-1]

    """     bugged version ?   
    delta_y = torch.tensor([-theta * (ys[i - 1] - mu) * DT + sigma * eta[i-1] for i in range(1, TS.size(0))])
    ys = torch.tensor([ys[i - 1] + delta_y[i-1] for i in range(1, TS.size(0))])
    """
    return TS[1:], ys[1:], delta_y[1:]
    
#def narma_N()

