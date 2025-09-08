### the function to compute similarity matrix ### 
import numpy as np
import pandas as pd
from dtaidistance import dtw
def dis_acc(X):
    # input：dataframe X
    # output： distance matrix
    n,d = X.shape
    X_1 = np.array(X)
    X_aver = np.mean(X_1,axis=0)
    acc = np.zeros((d,n-1))
    for i in range(d):
        X_s = np.sum((X_1[:,i]-X_aver[i])**2)
        for j in range(1,n):
            acc[i,j-1] = np.sum((X_1[range(j,n),i]-X_aver[i])*(X_1[range(n-j),i]-X_aver[i]))/X_s
    acc = np.sqrt(2*(1-acc))
    
    row_stds = np.std(acc, axis=1, keepdims=True)
    acc = acc / row_stds
    
    dis = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1,d):
            dis[j,i] = dis[i,j] = np.sqrt(np.dot(acc[i,:]-acc[j,:],(acc[i,:].T-acc[j,:].T))) 
    return dis

def dist_dtw_matrix(X):
    # input：dataframe X
    # output： distance matrix
    n,d = X.shape
    X_1 = np.array(X)
    X_aver = np.mean(X_1,axis=0)
    acc = np.zeros((d,n-1))
    for i in range(d):
        X_s = np.sum((X_1[:,i]-X_aver[i])**2)
        for j in range(1,n):
            acc[i,j-1] = np.sum((X_1[range(j,n),i]-X_aver[i])*(X_1[range(n-j),i]-X_aver[i]))/X_s
    acc = np.sqrt(2*(1-acc))
    row_stds = np.std(acc, axis=1, keepdims=True)
    acc = acc / row_stds
    dis =  dtw.distance_matrix(acc, parallel=True)
    return dis

def pearson_corr_dist(X):
    # input：dataframe X
    # output： distance matrix
    corr_mat = X.corr()
    dist = np.array(1-corr_mat)
    return dist


### spectral cluster based on GuassionKernal and sigma chosen by Silhouette Coefficient###
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
def spectral_cluster_kernal(X,n_cluster,dis=dis_acc,random_state=None):
    n,d = X.shape
    clusterAssment = np.zeros((d,2))
   
    dist_matrix = dis(X)
    
    # set the range of sigma
    median_dist = np.median(dist_matrix[np.triu_indices(d, k=1)])
    sigma_candidates = np.logspace(
        np.log10(0.1 * median_dist), 
        np.log10(10 * median_dist),
        15)
    
    best_sigma = None
    best_score = -1
    best_labels = None
    results = []
    
    for sigma in sigma_candidates:
        # use guassion kernal function
        sim_matrix = np.exp(-dist_matrix / (2 * sigma**2))
        np.fill_diagonal(sim_matrix, 0)
        spectral = SpectralClustering(n_clusters=n_cluster,affinity='precomputed',random_state=random_state,n_init=10)
        labels = spectral.fit_predict(sim_matrix)
        score = silhouette_score(dist_matrix,labels,metric='precomputed')
        results.append((sigma, score))
        if score > best_score:
            best_sigma = sigma
            best_score = score
            best_labels = labels
            
    # cluster result
    for i in range(len(best_labels)):
        clusterAssment[i,:]= [i,best_labels[i]]
    return clusterAssment,best_score

def spectral_cluster_plot(X,n_cluster,dis=dis_acc,random_state=None):
    n,d = X.shape
   
    dist_matrix = dis(X)
    
    # set the range of sigma
    median_dist = np.median(dist_matrix[np.triu_indices(d, k=1)])
    sigma_candidates = np.logspace(
        np.log10(0.1 * median_dist), 
        np.log10(10 * median_dist),
        15)
    
    results = np.zeros((15,2))
    j = 0
    for sigma in sigma_candidates:
        # use guassion kernal function
        sim_matrix = np.exp(-dist_matrix / (2 * sigma**2))
        np.fill_diagonal(sim_matrix, 0)
        spectral = SpectralClustering(n_clusters=n_cluster,affinity='precomputed',random_state=random_state,n_init=10)
        labels = spectral.fit_predict(sim_matrix)
        score = silhouette_score(dist_matrix,labels,metric='precomputed')
        results[j,:] = [sigma,score]
        j = j+1
    return results,sigma_candidates


        
    
    
   
        
    
    
   