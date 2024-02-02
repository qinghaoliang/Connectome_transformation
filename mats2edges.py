import numpy as np

def mats2edges(all_mats):
    num_node = np.shape(all_mats)[0]
    num_sub = np.shape(all_mats)[2]
    num_edges = num_node * (num_node-1) // 2

    all_edges = np.zeros([num_sub, num_edges])
    iu1 = np.triu_indices(num_node, 1)
    for i in range(num_sub):
        all_edges[i, :] = all_mats[iu1[0], iu1[1], i]

    return all_edges

def com2edges(mat):
    num_node = np.shape(mat)[0]
    iu1 = np.triu_indices(num_node, 1)
    edges = mat[iu1[0], iu1[1]]

    return edges

def nrmse(x_true, x_impute):
    return np.sqrt(np.mean((x_true-x_impute)**2)/np.var(x_true))

def mse(x_true, x_impute):
    return np.mean((x_true-x_impute)**2)

def remove_zero(data):
    val = np.sum(data, axis=0)
    return data[:, val>0]