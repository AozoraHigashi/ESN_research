import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from scipy.linalg import svd as svd
import math

FUNCTIONS = {"identity":(lambda x : x), "tanh":np.tanh}

class ESN_mult():
    def __init__(self,N,sigma=0.1,p=1,pin=1,rho=0.9,rseed=0,uC=0,dim=2,shiftreg=False,idWin=False):
        self.N,self.rho,self.sigma,self.p,self.pin,self.dim = N,rho,sigma,p,pin,dim
        # W : N * N 
        # Win : N * dim
        if shiftreg:
            # sum = 1 + ... + dim
            """ example <dim = 3, N = 600> 
            sum = 6
            unit = 100
            lengths = [100, 200, 300]
            """
            sum = dim*(dim+1)/2
            unit = int (N / sum)
            excess = N - unit*sum
            lengths =[unit*dim for dim in range(1,dim+1)]
            lengths[dim-1] = lengths[dim-1] + excess
            
            """
            print("should be ",N," : ",np.sum(lengths))
            print("should be ",dim, " : ", len(lengths))
            print("should be",N-1,":",len(d))            
            <Wについて>
            d : length-1 の1が並んだあと一つ0が入る、これの繰り返し
            W : d を対角の一つ下に並べた行列
            <Win>length ごとに1を入れる
            ただし、列も一つずつずれる
            """
            Win = np.zeros(shape=(N,dim))
            pos = 0
            for i in range(dim):
                Win[pos][i]=1
                pos += lengths[i]
            W = np.zeros(shape=(N,N))            
            d = np.ones(int(lengths[0]-1))
            for i in range(1,dim): 
                d = np.append(d,0)
                d = np.concatenate((d,np.ones(int(lengths[i]-1))),axis=None)   
            W = np.diag(d,k=-1)
            self.Win = Win
            self.W = W
            
        else : 
            np.random.seed(rseed)
            W = np.random.uniform(-1,1,(N,N))*(np.random.uniform(0,1,(N,N))<p)
            eigs = np.linalg.eigvals(W)
            self.W = rho*W/np.max(np.abs(eigs))
            if uC != 0 :
                U, s, Vh = svd(a=W)
                self.W = uC*(U @ np.identity(N) @ Vh)
            else :
                eigs = np.linalg.eigvals(W)
                self.W = rho*W/np.max(np.abs(eigs)) 
            np.random.seed(rseed+1)
            if idWin :
                Win = np.random.uniform(-sigma,sigma,N)*(np.random.uniform(0,1,N)<pin)
                self.Win = np.tile(Win,(dim,1)).T
            else: self.Win = np.random.uniform(-sigma,sigma,(N,dim))*(np.random.uniform(0,1,(N,dim))<pin)

    # run and washout in one
    def run_washout(self,u,Two,actf=None, f=np.tanh,X0=[]): 
        if actf != None : f = FUNCTIONS[actf]
        if len(X0)==0:
            X0 = np.ones(self.N)
        # u : dim * T
        if len(u[:,:1]) != self.dim :
            print("input dimension error: input must be shape (", self.dim,", T)")
        T = len(u[0])
        X = np.zeros((T,self.N))
        X[0] = X0
        for t in range(1,T):
            X[t] = f(self.W @ X[t-1] + self.Win@u[:,t-1])     #be careful with the expression
        
        Xwo = np.hstack([X[Two:],np.ones((T-Two,1))])
        return Xwo
    
# 多次元入力に対応したMC計算
def MCwithPI_general(u, Xwo, maxtau):       
        #try文を使って1次元の場合分け
        try : dim = len(u[:,:1])
        except IndexError : 
            dim = 1
            u = [u]
        T = len(Xwo)
        Two = len(u[0]) - T
        piX = np.linalg.pinv(Xwo)
        # Parameters for MC
        taus = np.arange(1, maxtau)
        mfs =[]
        for d in range(dim):
            # 行列計算による変換
            y_matrix = np.array([u[d][Two-tau:Two+T-tau] for tau in taus])
            mf_matrix = (y_matrix @ Xwo @ piX @ y_matrix.T)
            normalizer = np.diag(y_matrix @ y_matrix.T)
            mf = np.diag(mf_matrix)/normalizer
            mfs.append(mf)
        #print('MC',mc)
        return mfs