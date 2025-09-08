import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from numba import jit
from typing import Optional

from objective import Objective
from Build_lambdas import build_lambdas,max_lambdas

#软阈值函数
def soft_thresholding(x, b = 1.0):
    return np.sign(x) * np.maximum(np.abs(x) - b,0.)


def dual_scaling(XTtheta, lam_1,b = 1.):
    if b == 1.:
        dual_scale = np.maximum(lam_1, LA.norm(XTtheta, np.inf))
    else:
        omegaZ = max_lambdas(XTtheta,lam_1*b)
        dual_scale = np.maximum(lam_1,omegaZ)
    return dual_scale

def dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1,b=1.):
    pval = obj.val(screened_X, screened_w)
    pval += lam_1 * (b*LA.norm(screened_w, ord=1)+(1-b)*LA.norm(screened_w, ord=2))
    dval = obj.conj(lam_1 * theta / dual_scale)     
    gap = np.maximum(pval - dval,0.)
    return pval, dval, gap

def screening(w, XTcenter, r,norm2_X, n_active_features, disabled_features, b = 1.):
    n_features = w.shape[0]

    # Safe rule for Feature level
    for j in range(n_features):

        if disabled_features[j] == 1:  #1表示该资产已经被筛选出
            continue

        
        r_normX_j = r * np.sqrt(norm2_X[j]) 
        if r_normX_j > 1.:
            continue
            
        if b==1:

            if np.maximum(XTcenter[j], 0.) + r_normX_j < 1:
                w[j] = 0.
                disabled_features[j] = 1
                n_active_features -= 1
        else:
            if np.abs(XTcenter[j])+r_normX_j < b :
            #if max_lambda(XTcenter[j],b)+r *max_lambda(screened_X[:,j],b) < 1:
                w[j] = 0.
                disabled_features[j] = 1
                n_active_features -= 1
                
    return w, n_active_features    

@jit
def prox_gd(w, grad, L_dh, disabled_features, lam_1, w_old, w_old_old,b=1):
    n_features = w.shape[0]
    is_diff = False
    
    #l = np.linalg.norm(w)
    #if l == 0:
    #    thres = lam_1/L_dh
    #else:
    thres = lam_1*b / L_dh
    for j in range(n_features):
        if disabled_features[j] == 1:
            continue
        
        w_old_old[j] = w_old[j]
        w_old[j] = w[j]
        #if l == 0:
        #    grad_1 = grad[j]
        #else:
        grad_1  = grad[j] + lam_1*(1-b)*w[j]
            
       
        tmp = np.maximum(w[j] - (grad_1 / L_dh + thres), 0)
        #tmp = soft_thresholding(w[j]-grad_1 / L_dh, thres)
        if (w[j]-w_old_old[j])*(tmp-w[j])<0:
            if tmp-w[j]>0:
                w[j] = (w[j] + np.minimum(w_old_old[j], tmp)) / 2.0
            else:
                w[j] = (w[j] + np.maximum(w_old_old[j], tmp)) / 2.0
        elif tmp>w[j]>w_old_old[j]:
            w[j] = 2 * tmp - w[j]
        else:
            w[j] = tmp
        
        if is_diff is False and w[j]!=w_old[j]:
            is_diff = True        

    return w, is_diff 

@jit
def track_one_path(obj, X, w, theta0, norm2_X, L_dh,lam_1, max_iter, f, tol, screen,b):
    _, n_features = X.shape
    n_active_features = n_features
    
    final_pass = False
    gap = np.inf
    
    w_old = w.copy()
    w_old_old = w_old.copy()
    theta = theta0.copy()
    XTtheta = np.dot(X.T, theta)
    dual_scale = dual_scaling(XTtheta, lam_1,b)
    alpha = np.min(obj.comp_dual_neg_hess(theta0/dual_scale, lam_1))

    disabled_features = np.zeros(n_features)
    n_iter = 0
    for n_iter in range(max_iter):
        id_features = (disabled_features == 0)
        screened_w = w[id_features]
        screened_X = X[:, id_features]

        # Update dual variables
        theta[:] = obj.comp_dual_var(screened_X, screened_w)
        XTtheta[:] = np.dot(X.T, theta)
        #是否进行变量筛选
        if f != 0 and (n_iter % f == 0 or final_pass):   
            dual_scale = dual_scaling(XTtheta, lam_1,b)
            
            _, _, gap = dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1,b)

            if gap <= tol or final_pass:
                final_pass = True
                break

            if screen:
                r = np.sqrt(2 * gap / alpha)
                XTcenter = XTtheta / dual_scale
                
                w, n_active_features = screening(w,XTcenter, r,
                        norm2_X, n_active_features, disabled_features,b)
            
            # The local Lipschitz constant of h's gradient.
            L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2 + lam_1*(1-b)
            #L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2 

        if final_pass:
            break

        w, is_diff = prox_gd(w, -XTtheta, L_dh,
            disabled_features, lam_1, w_old, w_old_old,b)
        
        if not is_diff:
            final_pass = True

    w = obj.proj(w)

    return gap, n_active_features, n_iter


def spo_l1_path(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30,b:float = 1., verbose: bool = True):
    
    n_samples, n_features = X.shape
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features)  
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta,b)
        

    n_lambdas = lambdas.shape[0]

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2 
    
    ws = np.zeros((n_features, n_lambdas))
    gaps = np.ones(n_lambdas)
    n_active_features = np.zeros(n_lambdas)
    n_iters = np.zeros(n_lambdas)
    

    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,lambdas[t], max_iter, f, tol, screen,b)

        ws[:, t] = w_init.copy()
    return ws, lambdas, gaps, n_iters, n_active_features


def spo_nw_min(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, b:float = 1.,verbose: bool = True,
    nw_min=5):
    
    n_samples, n_features = X.shape
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta,b)

    n_lambdas = lambdas.shape[0]

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2 
    
    ws = np.zeros((n_features, n_lambdas))
    gaps = np.ones(n_lambdas)
    n_active_features = np.zeros(n_lambdas)
    n_iters = np.zeros(n_lambdas)
   

    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen,b)

        ws[:, t] = w_init.copy()
        if np.sum(w_init>0.)>=nw_min:
            break
    

    return ws, lambdas, gaps, n_iters, n_active_features
    