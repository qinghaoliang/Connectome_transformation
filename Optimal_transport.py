import numpy as np
import scipy.io as sio
import ot
import os
from tqdm import tqdm

def nrmse(x_true, x_impute):
    return np.sqrt(np.mean((x_true-x_impute)**2)/np.var(x_true))

def normalize(x):
    if all(v == 0 for v in x):
        return x
    else:
        if max(x) == min(x):
            return x
        else:
            return (x-min(x))/(max(x)-min(x))

def corr2_coeff(A, B):
	# Rowwise mean of input arrays & subtract from input arrays themeselves
	A_mA = A - A.mean(1)[:, None]
	B_mB = B - B.mean(1)[:, None]    # Sum of squares across rows
	ssA = (A_mA**2).sum(1)
	ssB = (B_mB**2).sum(1)
	return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def generate_correlation_map(x, y):
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must have the same number of timepoints.')
    s_x = x.std(1, ddof=n-1)+1e-6
    s_y = y.std(1, ddof=n-1)+1e-6
    cov = np.dot(x,y.T) - n * np.dot(mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def mapping(dataS, dataT, train_index, frame_size): 
    # learn the mapping between the node timeseries of two subjects 
    t = dataS.shape[2] 
    p1 = dataS.shape[1] 
    p2 = dataT.shape[1] 
    num_time_points = t

    train_size = len(train_index)
    # build cost matrix
    M = np.zeros((p1,p2))
    for i in train_index:
        a = dataS[i,:,:]
        b = dataT[i,:,:]
        Mi = generate_correlation_map(a,b)
        Mi = (Mi - np.min(Mi))/np.ptp(Mi)
        Mi = 1-Mi
        M = M + Mi
    M = M/train_size

    # optimal transport
    G = np.zeros((num_time_points,p1,p2))
    for j in range(0,num_time_points,frame_size):
        for i in train_index:
            lamda = 0.01
            a = dataS[i,:,j]
            b = dataT[i,:,j]
            a = normalize(a) + 1e-5
            b = normalize(b) + 1e-5
            a = a/np.sum(a)
            b = b/np.sum(b)
            T = ot.sinkhorn(a, b, M, lamda, numItermax=1000000, verbose=False)
            T = T/T.sum(axis=0, keepdims=1)
            G[j:j+frame_size] = G[j:j+frame_size]+ T
            
        G[j:j+frame_size] = G[j:j+frame_size]/train_size

    # the output mapping is the average across all timepoints
    G = G.mean(axis=0)
    G = G/G.sum(axis=0,keepdims=1)
    return G

def Optimal_transport(dataS, dataT):
    # learn the mapping on the dataset of n subjects with node timeseries
    #num_time_points = dataS.shape[2]
    nsub = np.shape(dataS)[0]
    dim_s = np.shape(dataS)[1]
    dim_t = np.shape(dataT)[1]
    T_ot = np.zeros((dim_s, dim_t, nsub))
    for i in range(nsub):
        train_index = [i]
        #G =  mapping(dataS, dataT, train_index, num_time_points)
        G =  mapping(dataS, dataT, train_index, 1)
        T_ot[:,:,i] = G
    return T_ot





    