import numpy as np
from numpy import linalg as LA


FUNC_LOG = 0
FUNC_EXP = 1
#求解h(w)
def _comp_val_LOG(X, w, eta):
    return - np.mean(np.log(np.dot(X, w) + eta))

#计算共轭函数的目标值
def _comp_conj_LOG(theta, eta):       
    n = theta.shape[0]
    return np.mean(np.log(n * theta) + 1 - n * eta * theta)  

#求解 λ*θ
def _comp_dual_var_LOG(X, w, eta):    
    return 1.0 / (np.dot(X, w) + eta) / X.shape[0]

#求解hessian矩阵  --- 用于函数的Lipschiz常数 求解
def _comp_dual_neg_hess_LOG(theta):   
    return 1. / theta**2 / theta.shape[0]

def _comp_val_EXP(X, w, eta, a):
    return -1.  + np.mean(np.exp(-(a * np.dot(X, w) + eta)))

def _comp_conj_EXP(theta, eta, a):
    n = theta.shape[0]
    return - n * np.mean(theta * (np.log(n * theta / a) - 1  + eta)) / a - 1.

def _comp_dual_var_EXP(X, w, eta, a):
    return a * np.exp(- (a * np.dot(X, w) + eta)) / X.shape[0]  

def _comp_dual_neg_hess_EXP(theta, a, lam):
    return lam / a / theta

#h(w)关于w 的梯度计算
def _comp_grad(X, theta):   
    return - np.dot(X.T, theta)

#结果处理
def _proj(w):    
    #c =  b*LA.norm(w,ord = 1) +(1-b)*LA.norm(w,ord = 2)
    #c = LA.norm(w,ord = 1) 
    c = np.sum(w)
    if c>0.:
        w[:] = w / c
    return w


class Objective(object):
    def __init__(self, n_samples, eta, func=FUNC_LOG, a:float = 1.):
        self.func = func        
        n_samples = n_samples
        if self.func == FUNC_LOG:  #对数效用函数
            self.a = 0.
            self.eta = eta
            self.L_u = 1./self.eta
            self.L_dH = 1./self.eta**2/n_samples
        
        else:      #指数效用函数
            self.a = a
            self.eta = eta
            self.L_u = self.a/np.exp(self.eta)
            self.L_dH = self.a**2/n_samples/np.exp(self.eta)
         
    #计算平均样本函数值
    def val(self, X, w):
        
        if self.func == FUNC_LOG:
            return _comp_val_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_val_EXP(X, w, self.eta, self.a)
    
    #计算梯度
    def grad(self, X, w):
        
        theta = self.comp_dual_var(w, X) 
        
        return _comp_grad(X, theta)
    
    #计算不同效用函数的共轭函数的值
    def conj(self, theta):
        
        if self.func == FUNC_LOG:
            return _comp_conj_LOG(theta, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_conj_EXP(theta, self.eta, self.a)
    
    #计算不同效用函数下的对偶问题的 theta*lam
    def comp_dual_var(self, X, w):  
       
        if self.func == FUNC_LOG:
            return _comp_dual_var_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_dual_var_EXP(X, w, self.eta, self.a)
    

    def proj(self, w):
        return _proj(w)

#求解GAP半径中的alpha需要使用    
    def comp_dual_neg_hess(self, theta, lam):
        
        if self.func == FUNC_LOG:
            return _comp_dual_neg_hess_LOG(theta)
        elif self.func == FUNC_EXP:
            return _comp_dual_neg_hess_EXP(theta, self.a, lam)