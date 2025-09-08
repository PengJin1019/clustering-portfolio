import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from dtaidistance import dtw
##### distance function ####
### two array ###
# compute distance of two array based on autocorrelation coefficient

def dis(x,y):
    #input : array x,y
    #output : distance between x and y based on autocorrelation coefficient
    n = len(x)
    X_aver = np.array([np.mean(x), np.mean(y)])
    X_1 = np.array([x,y])
    acc = np.zeros((2,n-1))
    for i in range(2):
        X_s = np.sum((X_1[i,:]-X_aver[i])**2)
        for j in range(1,n):
            acc[i,j-1] = np.sum((X_1[i,range(j,n)]-X_aver[i])*(X_1[i,range(n-j)]-X_aver[i]))/X_s
    acc = np.sqrt(2*(1-acc))
    row_stds = np.std(acc, axis=1, keepdims=True)
    acc = acc / row_stds
    dis = np.sqrt(np.dot(acc[0,:]-acc[1,:],(acc[0,:].T-acc[1,:].T))) 
    
    return dis

### matrix ###
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
            dis[i,j] = np.sqrt(np.dot(acc[i,:]-acc[j,:],(acc[i,:].T-acc[j,:].T))) 
            dis[j,i] = dis[i,j]
    return dis


### dis_dtw ###
def dis_dtw(x,y):
    #input : array x,y
    #output : distance between x and y based on autocorrelation coefficient
    n = len(x)
    X_aver = np.array([np.mean(x), np.mean(y)])
    X_1 = np.array([x,y])
    acc = np.zeros((2,n-1))
    for i in range(2):
        X_s = np.sum((X_1[i,:]-X_aver[i])**2)
        for j in range(1,n):
            acc[i,j-1] = np.sum((X_1[i,range(j,n)]-X_aver[i])*(X_1[i,range(n-j)]-X_aver[i]))/X_s
    acc = np.sqrt(2*(1-acc))
    row_stds = np.std(acc, axis=1, keepdims=True)
    acc = acc / row_stds
    dis =  dtw.distance_matrix(acc, parallel=True)[0,1]
    return dis

### matrix ###
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

### K-Means Cluster ###
#The choice of initial centroids
def cluster_choice(X,k,dis):
    #input: dateframe X
    #output: centroids
    X_0 = np.array(X)
    X_1 = np.transpose(X_0)
    #initial: random choose one column
    centroids = [X_1[np.random.choice(X_1.shape[0])]]
    #choose other centroids
    for i in range(1, k):
        distances = np.array([min([dis(x, c) for c in centroids]) for x in X_1])
        new_centroid_index = np.argmax(distances) 
        centroids.append(X_1[new_centroid_index])

    return pd.DataFrame(centroids).T


# Cluster
def kmeans(X,k,dis1=dis_acc,dis2=dis,max_iters=1000000, tol=1e-4):
    ### input ###
    #X : dataframe ,sample matrix
    #k : the number of clusters
    #dis1 : to compute the distance between stocks and centroids
    #dis2 : to compute distance of two array
    #max_inters : int，representing the maximum number of iterations, default is 10000
    #tol : the tolerance
    
    ### output ###
    # centroids, the label of cluster
    
    n,d = X.shape
    clusterAssment = np.zeros((d,2))
    #First column stores the cluster labels, and the second column stores the distances.
    
    #choose the initial centroids
    centroids = cluster_choice(X,k,dis2)
    
    L_name = ["cluster" + str(i) for i in range(1,k+1)] 
    centroids.columns = L_name
    centroids.index = X.index
    
    dist_matrix = dis1(X)

    for inter in range(max_iters):
        clusterAssment1 = clusterAssment.copy()
        X[L_name] = centroids
        distance  = dis1(X)
        
        for i in range(d):
            disji = [distance[i,d+j] for j in range(k)]
     
            # choose the nearest centroid
            clusterAssment[i,:] = [np.argmin(disji),np.min(disji)]
            
        # Re-COMPUTE centroids
        new_centroids = []
        for cent in range(k):
            t = np.where(clusterAssment[:,0]==cent)
            L = []
            for i in t[0]:
                L.append(i)
            p = X.iloc[:,L]
            new_centroids.append(np.mean(p,axis=1))
        new_centroids = pd.DataFrame(new_centroids).T
        new_centroids.columns = L_name
        
        #compute the gap between new and old centroids
        centroid_change = np.linalg.norm(np.array(new_centroids)-np.array(centroids))
     
        # update the centroids
        if new_centroids.isnull().any().any():
            clusterAssment = clusterAssment1
            break
        else:
            centroids = new_centroids
        # if gap < tol ,stop the iteration
        if centroid_change < tol :
            break
    X = X.drop(L_name,axis=1)

    labels = clusterAssment[:,0]
    score = silhouette_score(dist_matrix,labels,metric='precomputed')

    return X,centroids,clusterAssment,score