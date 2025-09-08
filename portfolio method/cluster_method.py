# %%
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
from kmeans import kmeans,dis_dtw
from spectral import spectral_cluster_kernal
from spectral import dis_acc,dist_dtw_matrix,pearson_corr_dist
import pandas as pd
import numpy as np
import os

# %%
df = pd.read_excel('df.xlsx',index_col=[0])
T, d = df.shape
df.index =  np.array(pd.to_datetime(df.index).strftime('%Y%m%d')).astype(int)
n_lambdas = 100
n_folds = 5
n_days_train = 120 #一只股票的数据量
n_days_hold = 63
id_begin = np.where(df.index>=20180103)[0][0]  #股票起始数据位置
df_hold = pd.read_excel('ret.xlsx', index_col=[0])+1
id_recal = np.arange(id_begin+120, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy() #特定行
df_hold = df_hold.fillna(1.)
n_recal = len(id_recal)
test_date = np.array(df.index[id_begin+120:id_recal[-2]+63])
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
    
df_or = pd.read_csv('LOG-1.00_original_weight-0.95.csv')
df_exp = pd.read_csv('EXP-1.00_original_weight-0.95.csv')
df_exp1 = pd.read_csv('EXP-1.50_original_weight-0.95.csv')

ret = pd.read_excel('ret.xlsx',index_col=[0])
ret.index =  np.array(pd.to_datetime(ret.index).strftime('%Y%m%d')).astype(int)

# %% [markdown]
# ## 1. cluster

# %% [markdown]
# ## kmeans-acc

# %%
path1 = 'KMEANS_new1/LOG'
path2 = 'KMEANS_new1/EXP1'
path3 = 'KMEANS_new1/EXP15'
os.makedirs(path1, exist_ok=True)
os.makedirs(path2, exist_ok=True)
os.makedirs(path3, exist_ok=True)

# %%
#LOG
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in range(12,22,1):
        L = []
        if(np.sum(df_or.iloc[:,tw]!=0)<=K_cluster):
            break
            #L = list(df.columns)
        else:
            for i,j in enumerate(df_or.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L] 
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
        X1,centroids,clusterAssment,score = kmeans(X1,K_cluster)
        
        if score > best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,0]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
    if(K is not None):
        filename =  'KMEANS_new1/LOG/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %%
# EXP-1.0
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in  range(12,22,1):
        L = []
        if(np.sum(df_exp.iloc[:,tw]!=0)<=K_cluster):
            # L = list(df.columns)
            break
        else:
            for i,j in enumerate(df_exp.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L] 
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
        X1,centroids,clusterAssment,score = kmeans(X1,K_cluster)
        
        if score > best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,0]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
     
    if(K is not None):
        filename = 'KMEANS_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %%
# EXP-1.5
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in  range(10,22,1):
        L = []
        if(np.sum(df_exp1.iloc[:,tw]!=0)<=K_cluster):
            # L = list(df.columns)
            break
        else:
            for i,j in enumerate(df_exp1.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L] 
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
        X1,centroids,clusterAssment,score = kmeans(X1,K_cluster)
        
        if score > best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,0]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
      
    if(K is not None):
        filename = 'KMEANS_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %% [markdown]
# ## spectral-acc

# %%
path1 = 'SPECTRAL_new/LOG'
path2 = 'SPECTRAL_new/EXP1'
path3 = 'SPECTRAL_new/EXP15'
os.makedirs(path1, exist_ok=True)
os.makedirs(path2, exist_ok=True)
os.makedirs(path3, exist_ok=True)

# %%
#LOG
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in  range(12,22,1):
        L = []
        if(np.sum(df_or.iloc[:,tw]!=0)<=K_cluster):
            #L = list(df.columns)
            break
        else:
            for i,j in enumerate(df_or.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
        
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L] 
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
        clusterAssment,score = spectral_cluster_kernal(X1,K_cluster,dis=dis_acc)
        
        if score >= best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,1]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
    if(K is not None):
        filename = 'SPECTRAL_new/LOG/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %%
#EXP-1.0
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in range(12,22,1):
        L = []
        if(np.sum(df_exp.iloc[:,tw]!=0)<=K_cluster):
            # L = list(df.columns)
            break
        else:
            for i,j in enumerate(df_exp.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
       
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L] 
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
     
        
        clusterAssment,score = spectral_cluster_kernal(X1,K_cluster,dis=dis_acc)
        
        if score >= best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,1]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
    
    if(K is not None):
        filename = 'SPECTRAL_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %%
#EXP-1.5
for tw in range(17):
    best_score = 0
    best_category = None
    best_k = 0
    K = None
    for K_cluster in  range(12,22,1):
        L = []
        if(np.sum(df_exp1.iloc[:,tw]!=0)<=K_cluster):
            # L = list(df.columns)
            break
        else:
            for i,j in enumerate(df_exp1.iloc[:,tw]):
                if j!=0:
                    L.append(df.columns[i])
       
        idx = id_recal[tw]
        X = df.iloc[idx-n_days_train:idx,:]
        X1 = X.loc[:,L]
        X1_ret = ret.iloc[idx-n_days_train:idx,:]
        X1_ret = X1_ret.loc[:,L]
        X_test = df.iloc[idx:idx+n_days_hold,:]
        X2 = X_test.loc[:,L]
     
        
        clusterAssment,score = spectral_cluster_kernal(X1,K_cluster,dis=dis_acc)
        
        if score >= best_score:
            best_k = K_cluster
            best_score = score
            best_category = clusterAssment[:,1]
            L = pd.DataFrame(np.array(L),columns=['stock'])
            best_L1 = pd.DataFrame(np.array(best_category),columns=['cluster'])
            K = pd.concat((L,best_L1),axis=1)
   
    if(K is not None):
        filename = 'SPECTRAL_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(best_k)+'时间窗口'+str(tw))
        K.to_csv(filename,index=False)
    print(1)
    

# %%


# %% [markdown]
# ## 2. Portfolio

# %% [markdown]
# ## 2.1 LOG

# %%
func_names = ['LOG', 'EXP']
func = 0
a = 1
screen = False
from joblib import Parallel, delayed

# %%
def fit2(i, df1,df2,func, a,screen):
    
    b_list = np.arange(0.8,1.01,0.02)
    i_len = 63
    d = df1.shape[1]
    X = np.array(df1.copy())
    X_test = np.array(df2.copy())
    score = np.zeros(len(b_list))
    ws_list = []
    score_list = []
    for k,b in enumerate(b_list):
        score_test = np.zeros((i_len,))
        ws_test = np.empty((1, d))
        ws_test.fill(np.nan)

        delta = 2.
        _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas,  delta=delta,
                                         max_iter=int(0),screen = False,b = b,verbose=False)

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
        s = np.median(score_val, axis=0)
        
        #id_lam = np.argmax(np.median(score_val, axis=0))   
        id_lam = np.where(s==np.max(s[s!=0]))[0][0]

        #id_lam = np.argmax(np.median(score_val, axis=0))    
        lambdas = lambdas[id_lam:id_lam+5]        
        ws, _, _, _, _ = spo_nw_min(X, func, a, lambdas, None,
                                    screen, max_iter=int(1e5), f=30,b = b, tol=1e-8, nw_min=1)
        id_lams = np.where(np.sum(ws>0, axis=0) > 0)[0] 
        
        if len(id_lams)>0:
           
            w = ws[:, id_lams[0]]
            score_test = np.dot(X_test, w) - 1
            ws_test[0, np.arange(0,d)] = w
            ret = np.log((X_test-1) @ w + 1)
        else:
            score_test = [0.]
            ws_test[0, np.arange(0,d)] = 0.
            ret = [0.]
        #sd = np.std(ret, axis=0)
        #score[k] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)
        score[k] = np.cumprod(np.asarray(score_test)+1.)[-1]
        ws_list.append(ws_test[0, np.arange(0,d)].copy())
        score_list.append(score_test.copy())

   
    t = np.argmax(score)
    b_g = np.array([b_list[t]])
    ws_test[0, np.arange(0,d)] = ws_list[t]
    score_test = score_list[t]
    
    return score_test, ws_test, lambdas,b_g

# %% [markdown]
# 1. SR

# %%
func_names = ['LOG', 'EXP']
func = 0
a = 1
screen = False
from joblib import Parallel, delayed

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_or.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/LOG/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
    
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2
        

    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/LOG/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/LOG/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_or.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]

    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/LOG/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]-1
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
        
      
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2
        
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/LOG/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/LOG/ret_return_ZH.csv',index=False)

# %% [markdown]
# ## 2.2 exp-1.0

# %%
func_names = ['LOG', 'EXP']
func = 1
a = 1
screen = False
from joblib import Parallel, delayed

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
      
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2
        
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/EXP1/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/EXP1/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
        
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2

    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/EXP1/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/EXP1/ret_return_ZH.csv',index=False)

# %% [markdown]
# ## 2.3 exp-1.5

# %%
func_names = ['LOG', 'EXP']
func = 1
a = 1.5
screen = False
from joblib import Parallel, delayed

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp1.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
 
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2
        
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/EXP15/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/EXP15/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp1.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'KMEANS_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
        
      
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2
        

    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('KMEANS_new/EXP15/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('KMEANS_new/EXP15/ret_return_ZH.csv',index=False)

# %% [markdown]
# ## 3. spectral

# %%
func_names = ['LOG', 'EXP']
func = 0
a = 1
screen = False
from joblib import Parallel, delayed

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_or.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/LOG/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
   
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2
        

    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/LOG/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/LOG/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_or.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/LOG/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
        
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2
        
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/LOG/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/LOG/ret_return_ZH.csv',index=False)

# %% [markdown]
# ## 3.2 exp-1.0

# %%
func_names = ['LOG', 'EXP']
func = 1
a = 1
screen = False

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
   
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2

    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/EXP1/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/EXP1/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/EXP1/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2
        
  
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/EXP1/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/EXP1/ret_return_ZH.csv',index=False)

# %% [markdown]
# ## 3.3 exp-1.5

# %%
func_names = ['LOG', 'EXP']
func = 1
a = 1.5
screen = False

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp1.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #Shaprio-ratio
        df_ratio = pd.DataFrame()
        for n in range(k_num):
            s =  np.std(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0)
            k = np.divide(np.mean(X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])], axis=0), s, out=np.zeros_like(s), where=s != 0)
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_ratio = pd.concat((df_ratio,sr_top),axis=1)
    
      
        df_ratio_test = X2.loc[:,np.array(df_ratio.columns)]
    else:
        df_ratio = X1
        df_ratio_test = X2
        
  
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_ratio ,df_ratio_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/EXP15/sr_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/EXP15/sr_return_ZH.csv',index=False)

# %%
ws_sr_log = pd.DataFrame()
score_sr_log = pd.DataFrame()
for tw in range(17):
    w = None
    s1 = None
    L = []
    for i,j in enumerate(df_exp1.iloc[:,tw]):
        if j!=0:
            L.append(df.columns[i])
    idx = id_recal[tw]
    X = df.iloc[idx-n_days_train:idx,:]
    X1 = X.loc[:,L] 
    X1_ret = ret.iloc[idx-n_days_train:idx,:]
    X1_ret = X1_ret.loc[:,L]
    X_test = df_hold.iloc[idx:idx+n_days_hold,:]
    X2 = X_test.loc[:,L]
    for k_cluster in range(12,22,1):
        filename = 'SPECTRAL_new/EXP15/clsuter_{}.csv'.format('聚类数目'+str(k_cluster)+'时间窗口'+str(tw))
        if os.path.exists(filename):
            k_num = k_cluster
            break

    if os.path.exists(filename):
        K = pd.read_csv(filename)
        
        #return
        df_return = pd.DataFrame()
        for n in range(k_num):
            k = np.cumprod((X1_ret.loc[:,np.array(K[K.loc[:,'cluster']==n].iloc[:,0])]+1)).iloc[-1,:]
            if (len(k)!=0):
                sr_top = X1.loc[:,k.index[np.argmax(k)]]
                df_return = pd.concat((df_return,sr_top),axis=1)
    
    
        df_return_test= X2.loc[:,np.array(df_return.columns)]
    else:
        df_return = X1
        df_return_test = X2
        
 
    with Parallel(n_jobs=-1, verbose=100) as parallel:
        out = parallel(delayed(fit2)(i, df_return ,df_return_test, func, a,screen) for i in range(0,1)) 
        score_test_list, ws_test_list, lambdas,b_g = zip(*out)
        score_test_list= np.concatenate(score_test_list, axis=0)
        ws_test_list= np.concatenate(ws_test_list, axis=0)
        lambdas = np.concatenate(lambdas, axis=0)
        b_g= np.concatenate(b_g, axis=0)
            
    
    w = pd.DataFrame(ws_test_list[0],columns=[str(tw)])
    s1 = pd.DataFrame(score_test_list,columns=[str(tw)])
    ws_sr_log = pd.concat((ws_sr_log,w),axis=1)
    score_sr_log = pd.concat((score_sr_log,s1),axis=1)
    
ws_sr_log.to_csv('SPECTRAL_new/EXP15/ret_weight_ZH.csv',index=False)
score_sr_log.to_csv('SPECTRAL_new/EXP15/ret_return_ZH.csv',index=False)

# %%



