import numpy as np
import scipy.io as sio
import ot
import os

def load_data_sc(atlas, task):
    #### load the structure connectomes ####
    path = './data/sc/'
    fname = path + task + "_" + atlas + ".mat"
    # the data is in shape (node, node, nsub)
    data = sio.loadmat(fname) 
    x = data['mats']
    print(atlas,task,x.shape)
    return x

def load_data_fc(atlas, task):
    #### load the functional connectomes ####
    path = './data/fc/'
    fname = path + task + "_" + atlas + ".mat"
    # the data is in shape (node, node, nsub)
    data = sio.loadmat(fname)
    x = data['mats']
    print(atlas,task,x.shape)
    return x

def load_data_node(atlas, task):
    #### load the node timeseries ####
    path = './data/node_time/'
    fname = path + atlas + "_" + task + ".mat"
    # the data is in shape (nsub, node, ntime)
    data = sio.loadmat(fname)
    x = data['mats']

    print(atlas,task,x.shape)
    
    return x

def load_data_mapping_time(atlas):
    #### load the node timeseries data for mapping ####
    path = './data/mapping_time/'
    fname = path + atlas + ".mat"
    # the data is in shape (nsub, node, ntime)
    data = sio.loadmat(fname)
    x = data['mats']

    print(atlas,x.shape)
    
    return x

def load_data_mapping_fc(atlas):
    #### load the node fc data for mapping ####
    path = './data/mapping_fc/'
    fname = path + atlas + ".mat"
    # the data is in shape (nsub, node, ntime)
    data = sio.loadmat(fname)
    x = data['mats']

    print(atlas,x.shape)
    
    return x

def load_data_mapping_sc(atlas, task):
    #### load the node sc data for mapping ####
    path = './data/mapping_sc/'
    fname = path + atlas + "_" + task + ".mat"
    # the data is in shape (nsub, node, ntime)
    data = sio.loadmat(fname)
    x = data['mats']

    print(atlas,task,x.shape)
    
    return x

