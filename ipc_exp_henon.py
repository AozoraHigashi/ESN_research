import torch
import matplotlib.pyplot as plt
import ESN
import time
import numpy as np
import json
import csv
from dataclasses import dataclass, asdict

torch.set_default_device("cuda:0")
torch.set_default_dtype(torch.double)

# Parameters
Two,Ttrain = 2000,100000
N = 10
C = 0
rho=0.9
dim = 2

#sigma = 0.5
sigmas = torch.linspace(0.1,2.0,39)

N_d = int(N * dim)
shftreg = False
idwin = False
#actf = "identity"
actf = "tanh"


fn_i = "10N_2din"
u_sym = torch.load('./experiments/inputs/'+fn_i+'_i.pt')
ti = torch.load('./experiments/target_info/10N_2din_ti.pt')


sigmas = sigmas[:3]

for sigma in sigmas:
    
    fn = r"henon_%.2f_s"%(sigma)
    
    ## construct Henon map model
    ## run and washout 
    
    henon = Dynamics_Res.Henon()
    st = time.time()
    Xwo = henon.run_washout(u_sym, Two)
    print("runtime :",time.time()-st)

    torch.save(Xwo,f"./experiments/datamatrices/{fn}_d.pt")

    ## calculate ipc
    
    st = time.time()
    
    raw,lin,rev,sur = ESN.calc_capacity(Xwo,targets,ret_all=True)
    
    print("ipc :",time.time()-st)
    capacities = rev
    torch.save(capacities,f"./experiments/ipcs/{fn}_c.pt")
    torch.save(raw,f"./experiments/ipcs/{fn}_raw.pt")
    torch.save(sur,f"./experiments/ipcs/{fn}_sur.pt")
    
    print("result for sigma = ",sigma)
       
    
    print("totoal capacity:",totCapacity)
    print("MC:",totMC)