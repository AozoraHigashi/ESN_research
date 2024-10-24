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
N = 100
C = 0.9
rho=0.9
dim = 1

#sigma = 0.5
sigmas = torch.linspace(0.1,2.0,39)



N_d = int(N * dim)

maxtau = int(N_d / dim * 2)
#maxtau = int(N_d *1.5)
taus = np.arange(1,maxtau)

shftreg = False
idwin = False
#actf = "identity"
actf = "tanh"



fn_i = "100N_tanh"
u_sym = torch.load('./experiments/inputs/'+fn_i+'_i.pt')

## set max delay, degree
maxdd=[[2,30],[3,20],[4,10],[5,6]]

st = time.time()
target_info = ESN.make_targets(u_sym,maxdd,Two=Two)

targets = target_info.tar_f
dgrs = target_info.degree
print("target :",time.time()-st)


# Save to CSV
with open(f'./experiments/target_info/sigma_ipc.csv', mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=target_info.__annotations__.keys())
    writer.writeheader()
    writer.writerow(asdict(target_info))



sigmas = sigmas[:3]
for sigma in sigmas:
    fn = r"100N_%.2f s"%(sigma)

    setting = {"input dim":dim,"Two":Two, "Ttrain":Ttrain,"sigma":float(sigma),
                "Nodes":N_d,"uC":C, "actf":actf,"identical Win":idwin,"input dist":"uniform"}
    # store experiment setting 
    with open('./experiments/settings/'+fn+'_s.txt', 'x') as fp:
        data = json.dump(setting,fp)

    ## construct ESN model
    ## run and washout 
    esn = ESN.ESN_mult(N_d, uC=C, dim=dim,idWin = idwin,sigma=sigma)
    st = time.time()
    Xwo = esn.run_washout(u_sym, Two, actf=actf)
    print("runtime :",time.time()-st)

    torch.save(Xwo,f"./experiments/datamatrices/{fn}_d.pt")
    ## calculate memory capacity
    N = setting["Nodes"]
    dim = setting["input dim"]
    if setting["identical Win"]:
        maxtau = int(N*1.5) 
    else : maxtau =  int(N/dim *2) 


    st = time.time()
    #mfs = ESN.MCwithPI_general(u_sym, Xwo, maxtau)
    #mfs = ESN.MCwithPI_general_newsur(u_sym, Xwo, maxtau,sur_sets=1)
    #mfs = ESN.MC_cSVD_old(u_sym, Xwo, maxtau)
    mfs = ESN.MC_cSVD(u_sym, Xwo, maxtau)
    print("mc calc time :",time.time()-st)
    print("total MC :",float(torch.sum(mfs)))
    torch.save(mfs,f"./experiments/mfs/{fn}_mf.pt")
    ## calculate ipc
    
    st = time.time()
    raw,lin,rev,sur = ESN.calc_capacity(Xwo,targets,ret_all=True)
    print("ipc :",time.time()-st)
    capacities = rev
    torch.save(capacities,f"./experiments/ipcs/{fn}_c.pt")
    torch.save(raw,f"./experiments/ipcs/{fn}_raw.pt")
    torch.save(sur,f"./experiments/ipcs/{fn}_sur.pt")
    #print(sur)
    #print(torch.mean(raw))
    #print(torch.mean(lin))
    totMC=float(torch.sum(mfs))
    totCapacity = totMC+float(torch.sum(capacities))
    print("result for sigma = ",sigma)
    print("totoal capacity:",totCapacity)
    print("MC:",totMC)