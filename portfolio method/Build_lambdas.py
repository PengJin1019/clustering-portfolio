import numpy as np
from numba import jit
from numpy import linalg as LA
import sympy as sp

@jit(nopython=True, fastmath=True)
def max_lambdas(XTtheta0, b):
    s = 0
    for c in XTtheta0:
        abs_c = abs(c)
        tmp = abs_c / b
        if tmp > 0:
            for x in np.linspace(0, tmp, 1000):
                if x <= tmp:
                    expr1 = (np.sign(c) * (abs_c - b * x)) ** 2
                    s += expr1
        s -= (1 - b) * tmp ** 2
    x = np.sqrt(s / len(XTtheta0))
    return x

@jit(nopython=True, fastmath=True)
def max_lambda(XTtheta0, b):
    loss = list()
    s = 0
    L = np.arange(0.1, 100, 0.001)
    for v in L:
        s = np.sum(np.maximum(0, np.abs(XTtheta0) - b * v) ** 2) - ((1-b) * v) ** 2
        loss.append(np.abs(s))
    loss = np.array(loss)
    return L[np.argmin(loss)]

    

#生成一系列lambda值
def build_lambdas(XTtheta0, n_lambdas=10, delta=2.0,b=1.0):
    if b==1:
        lambda_max = LA.norm(XTtheta0, np.inf)
    else:
        lambda_max = max_lambda(XTtheta0, b)
        
    
    if n_lambdas == 1:
        lambdas = np.array([lambda_max])
        
    else:        
        lambdas = np.power(10,
            np.linspace(
                np.log10(lambda_max), np.log10(lambda_max)-delta, n_lambdas
                ))
        
    return lambdas
