import numpy as np
from corr_pval import corr_pval
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import f_regression, f_classif
from ot import gromov_wasserstein, unif
from mats2edges import mats2edges, com2edges
from utils import load_data_sc, load_data_fc
from spectral_embedding import spectral_embedding
from scipy.stats import pearsonr
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

def CPM(source, modality, model, all_behav, seed, 
            n_splits = 5, thresh=0.1, alphas_ridge=10**np.linspace(2, -5, 10)):
    # all_behav should be array of size (n,)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if modality == "fc":
        task = "wm"
        com = load_data_fc(source, task)
        id = ~np.isnan(all_behav)
        all_behav = all_behav[id]
        com = com[:,:,id]
        #index = np.load("index_fc.npy")
    elif modality == "sc":
        task = "qa_pass"
        com = load_data_sc(source, task)
        #index = np.load("index_sc.npy")

    nsub = np.shape(com)[2]
    nnode = np.shape(com)[0]

    k = 0
    all_edges = mats2edges(com)
    #all_edges = all_edges[index,:]

    nsub = np.shape(all_edges)[0]
    r2 = np.zeros(n_splits)
    acc = np.zeros(n_splits)
    pred = np.zeros(nsub)
    for train_index, test_index in kf.split(all_edges):
        # split the data into training and test
        x_test, y_test = all_edges[test_index], all_behav[test_index]
        x_train, y_train = all_edges[train_index], all_behav[train_index]
        
        # feature selection
        if model == "ridge":
            #_, edges_p = corr_pval(x_train, y_train)
            #edges_select = edges_p < thresh 
            #x_train = x_train[:, edges_select]
            #x_test = x_test[:, edges_select]
            ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
            ridge_grid.fit(x_train, y_train)
            r2[k] = ridge_grid.score(x_test, y_test)
            pred[test_index] = ridge_grid.predict(x_test)

        elif model == "svm":
            #_, edges_p = f_classif(x_train, y_train)
            #edges_select = edges_p < thresh 
            #x_train = x_train[:, edges_select]
            #x_test = x_test[:, edges_select]
            svc_clf = LinearSVC()
            svc_clf.fit(x_train, y_train)
            acc[k] = svc_clf.score(x_test, y_test)

        #update the iteration counts
        k=k+1

    if model == "ridge":  
        #r2[r2 < 0] = 0
        #r = np.mean(np.sqrt(r2))
        r, p = pearsonr(pred, all_behav)
        return r, p
    elif model == "svm":
        acc = np.mean(acc)
        return acc 
    

def cpm_ot(source, target, modality, model, trans_set, all_behav, seed, 
            n_splits = 5, thresh=0.1, alphas_ridge=10**np.linspace(2, -5, 10)):
    # all_behav should be array of size (n,)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if modality == "fc":
        fname = "./map/map_fc_" + source + "_" + target + ".npz"
        mapping = np.load(fname)
        Trans = mapping['Trans']
        task = "wm"
        comS = load_data_fc(source, task)
        comT = load_data_fc(target, task)
        id = ~np.isnan(all_behav)
        all_behav = all_behav[id]
        comS = comS[:,:,id]
        comT = comT[:,:,id]
        #index = np.load("index_fc.npy")
    elif modality == "sc":
        fname = "./map/map_sc_" + source + "_" + target + ".npz"
        mapping = np.load(fname)
        Trans = mapping['Trans']
        task = "qa_pass"
        comS = load_data_sc(source, task)
        comT = load_data_sc(target, task)
        #index = np.load("index_sc.npy")
    elif modality == "sc_fcmap":
        fname = "./map/map_fc_" + source + "_" + target + ".npz"
        mapping = np.load(fname)
        Trans = mapping['Trans']
        task = "qa_pass"
        comS = load_data_sc(source, task)
        comT = load_data_sc(target, task)
        #index = np.load("index_sc.npy")
    elif modality == "fc_scmap":
        fname = "./map/map_sc_" + source + "_" + target + ".npz"
        mapping = np.load(fname)
        Trans = mapping['Trans']
        task = "wm"
        comS = load_data_fc(source, task)
        comT = load_data_fc(target, task)
        id = ~np.isnan(all_behav)
        all_behav = all_behav[id]
        comS = comS[:,:,id]
        comT = comT[:,:,id]

    nsub = np.shape(comS)[2]
    nnode = np.shape(comT)[0]
    comR = np.zeros((nnode, nnode, nsub))
    dim_t = np.shape(comT)[1]
    for i in range(nsub):
        com_s = comS[:,:,i]
        com_t = comT[:,:,i]
        AT = com_t
        AS = com_s
        Xp, Xn = spectral_embedding(AS, p=50)
        if modality == "sc" or modality == "fc_scmap":
            Xtp = dim_t*Trans.T@Xp
            Xtn = dim_t*Trans.T@Xn
        else:
            Xtp = Trans.T@Xp
            Xtn = Trans.T@Xn
        ATp = Xtp@Xtp.T
        ATn = Xtn@Xtn.T
        Ar = ATp-ATn
        comR[:,:,i] = Ar

    k = 0
    all_edgesS = mats2edges(comS)
    all_edgesT = mats2edges(comT)
    all_edgesR = mats2edges(comR)
    nsub = np.shape(all_edgesT)[0]
    r2 = np.zeros(n_splits)
    acc = np.zeros(n_splits)
    pred = np.zeros(nsub)
    for train_index, test_index in kf.split(all_edgesS):
        # split the data into training and test
        if trans_set == "train":
            x_test, y_test = all_edgesT[test_index], all_behav[test_index]
            x_train, y_train = all_edgesR[train_index], all_behav[train_index]
        elif trans_set == "test":
            x_test, y_test = all_edgesR[test_index], all_behav[test_index]
            x_train, y_train = all_edgesT[train_index], all_behav[train_index]
        
        # feature selection
        if model == "ridge":
            #_, edges_p = corr_pval(x_train, y_train)
            #edges_select = edges_p < thresh 
            #x_train = x_train[:, edges_select]
            #x_test = x_test[:, edges_select]
            ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
            ridge_grid.fit(x_train, y_train)
            r2[k] = ridge_grid.score(x_test, y_test)
            pred[test_index] = ridge_grid.predict(x_test)

        elif model == "svm":
            #_, edges_p = f_classif(x_train, y_train)
            #edges_select = edges_p < thresh 
            #x_train = x_train[:, edges_select]
            #x_test = x_test[:, edges_select]
            svc_clf = LinearSVC()
            svc_clf.fit(x_train, y_train)
            acc[k] = svc_clf.score(x_test, y_test)

        #update the iteration counts
        k=k+1

    if model == "ridge":  
        #r2[r2 < 0] = 0
        #r = np.mean(np.sqrt(r2))
        r, p = pearsonr(pred, all_behav)
        return r, p
    elif model == "svm":
        acc = np.mean(acc)
        return acc 
    

def cpm_stack(target, modality, model, all_behav, seed, 
            n_splits = 5, thresh=0.1, alphas_ridge=10**np.linspace(2, -5, 10)):
    # all_behav should be array of size (n,)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    p = 50
    atlas = ["shen", "craddock", "dosenbach", "schaefer", "brainnetome"]
    nnodes = {"shen":268, "craddock":200, "dosenbach":160, "schaefer":400, "brainnetome":246}
    atlas.remove(target)
    m = len(atlas)
    nnode = nnodes[target]
    id = ~np.isnan(all_behav)
    all_behav = all_behav[id]
    nsub = len(all_behav) 
    comR_all = np.zeros((nnode,nnode,nsub,m))
    for k in range(m):
        source = atlas[k]
        if modality == "fc":
            fname = "./map/map_fc_" + source + "_" + target + ".npz"
            mapping = np.load(fname)
            Trans = mapping['Trans']
            task = "wm"
            comS = load_data_fc(source, task)
            comT = load_data_fc(target, task)
            comS = comS[:,:,id]
            comT = comT[:,:,id]
        elif modality == "sc_fcmap":
            fname = "./map/map_fc_" + source + "_" + target + ".npz"
            mapping = np.load(fname)
            Trans = mapping['Trans']
            task = "qa_pass"
            comS = load_data_sc(source, task)
            comT = load_data_sc(target, task)
            comS = comS[:,:,id]
            comT = comT[:,:,id]
   
        nnode = np.shape(comT)[0]
        dim_t = np.shape(comT)[1]
        # reconstruction accuracy
        for i in range(nsub):
            com_s = comS[:,:,i]
            com_t = comT[:,:,i]
            AT = com_t
            AS = com_s
            Xp, Xn = spectral_embedding(AS, p=p) 
            Xtp = dim_t*Trans.T@Xp
            Xtn = dim_t*Trans.T@Xn
            ATp = Xtp@Xtp.T
            ATn = Xtn@Xtn.T
            Ar = ATp-ATn
            comR_all[:,:,i,k] = Ar
            
    rot = np.zeros(nsub)
    pot = np.zeros(nsub)
    comR = np.mean(comR_all, axis=3)
    for i in range(nsub):
        AT = comT[:,:,i]
        Ar = comR[:,:,i] 
        rot[i], pot[i] = pearsonr(com2edges(AT), com2edges(Ar))
    print(np.mean(rot, axis=0))

    k = 0
    all_edgesS = mats2edges(comS)
    all_edgesT = mats2edges(comT)
    all_edgesR = mats2edges(comR)
    nsub = np.shape(all_edgesT)[0]
    r2 = np.zeros(n_splits)
    acc = np.zeros(n_splits)
    pred = np.zeros(nsub)
    for train_index, test_index in kf.split(all_edgesS):
        # split the data into training and test
        x_test, y_test = all_edgesT[test_index], all_behav[test_index]
        x_train, y_train = all_edgesR[train_index], all_behav[train_index]
        
        # feature selection
        if model == "ridge":
            ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
            ridge_grid.fit(x_train, y_train)
            r2[k] = ridge_grid.score(x_test, y_test)
            pred[test_index] = ridge_grid.predict(x_test)
        elif model == "svm":
            svc_clf = LinearSVC()
            svc_clf.fit(x_train, y_train)
            acc[k] = svc_clf.score(x_test, y_test)

        #update the iteration counts
        k=k+1

    if model == "ridge":  
        r = pearsonr(pred, all_behav)[0]
        print(r)
        return r
    elif model == "svm":
        acc = np.mean(acc)
        return acc 
    


