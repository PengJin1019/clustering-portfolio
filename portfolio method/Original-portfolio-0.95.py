from numpy import linalg as LA
FUNC_LOG = 0
FUNC_EXP = 1
import numba as nb
from  Build_lambdas import build_lambdas
from objective import Objective
from new_spo import spo_l1_path,spo_nw_min
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

df = pd.read_excel('df.xlsx',index_col=[0])
T, d = df.shape
df.index =  np.array(pd.to_datetime(df.index).strftime('%Y%m%d')).astype(int)

n_lambdas = 100
n_folds = 5
n_days_train = 120 
n_days_hold = 63


id_begin = np.where(df.index>=20180103)[0][0]  #股票起始数据位置
df_hold = pd.read_excel('ret.xlsx', index_col=[0])+1
id_recal = np.arange(id_begin+120, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy() #特定行
df_hold = df_hold.fillna(1.)
n_recal = len(id_recal)-2
test_date = np.array(df.index[id_begin+120:id_recal[-3]+63])

df_listed = pd.read_excel('sp_list.xlsx', index_col=[0])  
df_listed.index = np.array(pd.to_datetime(df_listed.index).strftime('%Y%m%d')).astype(int)
id_codes_list = []
for idx in id_recal:
    codes = df_listed.columns[
        (np.all(df_listed[
            (df_listed.index>=df.index[idx-n_days_hold*4])&
            (df_listed.index<=df.index[idx])]==1, axis=0))]
    codes = codes[~df.iloc[idx-n_days_train:idx,:].loc[:,codes].isnull().any()]
    id_codes_list.append(
        np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
        )

def fit(i, df,func, a,screen):
    b_list = [0.95]
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]
    X = df.iloc[idx-n_days_train:idx,:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values[:,id_codes_list[i]].copy()
    score = np.zeros(len(b_list))  #保存每个b值求解得出的收益率
    ws_list = []
    score_list = []
    for k,b in enumerate(b_list):
        score_test = np.zeros((i_len,))
        ws_test = np.empty((1, d))
        ws_test.fill(np.nan)

    
    
        delta = 2.
        _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas,  delta=delta,
                                         max_iter=int(0),screen = screen,b = b,verbose=False)

        kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)
        score_val = np.zeros((n_folds, n_lambdas))
        for j, (train_index, val_index) in tqdm(enumerate(kf.split(X)), total = n_folds):
            X_train, X_val = X[train_index], X[val_index]

            ws, _, _, _, _ = spo_l1_path(X_train, func, a, lambdas, None, 
                                         screen, max_iter=int(1e4), f=30,b=b,tol=1e-5, verbose=False)

        
            # by sharpe ratio
            ret = np.log((X_val-1) @ ws + 1)
            sd = np.std(ret, axis=0)
            score_val[j] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)

        id_lam = np.argmax(np.median(score_val, axis=0))    
        lambdas = lambdas[id_lam:id_lam+5]        
        ws, _, _, _, _ = spo_nw_min(X, func, a, lambdas, None,
                                    screen, max_iter=int(1e5), f=30,b = b, tol=1e-8, nw_min=1)
        id_lams = np.where(np.sum(ws>0, axis=0) > 0)[0]  
        
        if len(id_lams)>0:
           
            w = ws[:, id_lams[0]]
            score_test = np.dot(X_test, w) - 1
            ws_test[0, id_codes_list[i]] = w
            ret = np.log((X_test-1) @ w + 1)
        else:
            score_test = [0.]
            ws_test[0, id_codes_list[i]] = 0.
            ret = [0.]
        sd = np.std(ret, axis=0)
        score[k] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)
        ws_list.append(ws_test[0, id_codes_list[i]].copy())
        score_list.append(score_test.copy()) 
    
    t = np.argmax(score)
    b_g = np.array([b_list[t]])
    ws_test[0, id_codes_list[i]] = ws_list[t]
    score_test = score_list[t]
    
    return score_test, ws_test, lambdas,b_g

func_names = ['LOG', 'EXP']
#LOG
func = 0
a = 1
screen = True
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a,screen) for i in range(len(id_recal)-2) )
    score_test_list, ws_test_list, lambdas,b_g = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    b_g = np.concatenate(b_g, axis=0)

df_or = pd.DataFrame(len(id_recal)-2)
for i in range():
    w = ws_test_list[i]
    if(np.sum(w)!=0):
        w = pd.DataFrame(w,columns=[str(i)])
        df_or = pd.concat((df_or,w),axis=1)   
df_or.to_csv('LOG-1.00_original_weight-0.95.csv',index=False)


#EXP-1.0
func = 1
a = 1
screen = True
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a,screen) for i in range(len(id_recal)-2) )
    score_test_list, ws_test_list, lambdas,b_g = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    b_g = np.concatenate(b_g, axis=0)

df_or = pd.DataFrame(len(id_recal)-2)
for i in range():
    w = ws_test_list[i]
    if(np.sum(w)!=0):
        w = pd.DataFrame(w,columns=[str(i)])
        df_or = pd.concat((df_or,w),axis=1)   
df_or.to_csv('EXP-1.00_original_weight-0.95.csv',index=False)


#EXP-1.5
func = 1
a = 1.5
screen = True
from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a,screen) for i in range(len(id_recal)-2) )
    score_test_list, ws_test_list, lambdas,b_g = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    b_g = np.concatenate(b_g, axis=0)

df_or = pd.DataFrame(len(id_recal)-2)
for i in range():
    w = ws_test_list[i]
    if(np.sum(w)!=0):
        w = pd.DataFrame(w,columns=[str(i)])
        df_or = pd.concat((df_or,w),axis=1)   
df_or.to_csv('EXP-1.50_original_weight-0.95.csv',index=False)

