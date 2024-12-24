import numpy as np
import torch

class Henon():
    def __init__(self, a=0.35, b=0.3, sigma=0.1):
        self.N=2
        self.dim=2
        self.sigma = sigma
        
        def func1(arg1,arg2):
            return 1 - a*arg1+arg2
        def func2(arg1):
            return b*arg1
        self.x1 = func1
        self.x2 = func2
        return
        
    def run_washout(self, u, Two, X0=None): 
        T = u.shape[1]
        X = torch.zeros((T, self.N))
        if X0 != None : X[0] = X0
        
        for t in range(1,T) :
            v1 = X[t-1][0] + self.sigma*u[0][t-1]
            v2 = X[t-1][1] + self.sigma*u[1][t-1]
            
            X[t][0] = self.x1(v1,v2)
            X[t][1] = self.x2(v2)
        
        Xwo = X[Two:]
        
        return Xwo

    
## saturation function for logistic map reservoir
def saturation_func(x:torch.tensor) -> torch.tensor: 
    a_0 = (x>0)*x
    a_1 = a_0>1
    res = a_0 - a_0*a_1 + a_1*1
    return res
    
    
class Logistic():
    def __init__(self,dim=2, rho=0.9, sigma=0.1):
        self.N=dim
        self.dim=dim
        self.sigma = sigma
        self.rho = rho
        def logi_func(x):
            return self.rho*x*(1-x)
        return
    
    def run_washout(self, u, Two, X0=None): 
        T = u.shape[1]
        X = torch.zeros((T, self.N))
        if X0 != None : X[0] = X0
        
        for t in range(1,T) :
            a = X[t-1]
            b = self.sigma * u[:,t-1]
            v = saturation_func(X[t-1] + self.sigma * u[:,t-1])
            X[t] = self.rho * v * (1-v)
        
        Xwo = X[Two:]
        
        return Xwo

    
    
    """"
class Lotka_Volterra():
    def __init__(self,  PARAMETERS  ,sigma=0.1):
        
    def run_washout(self, u, Two, X0=None): 
        if X0 != None : X[0] = X0
        
        T = u.shape[1]
        
        for t in range(1,T) :
            aa
       
        Xwo = X[Two:]
        
        return Xwo
"""