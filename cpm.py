import numpy as np
from corr_pval import corr_pval
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, roc_auc_score

def cpm_reg(all_edges, all_behav, seed, thresh=0.1, 
            alphas_ridge=10**np.linspace(2, -5, 10)):
    """Connectome-based Predictive Modeling for regression"""

    # all_behav should be array of size (n,)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_ridge = np.zeros(n_splits)
    k = 0
    for train_index, test_index in kf.split(all_edges):
        # split the data into training and test
        x_test, y_test = all_edges[test_index], all_behav[test_index]
        x_train, y_train = all_edges[train_index], all_behav[train_index]
        
        # feature selection
        edges_corr, edges_p = corr_pval(x_train, y_train)
        edges_select = edges_p < thresh 
        x_train = x_train[:, edges_select]
        x_test = x_test[:, edges_select]
        edges_corr = edges_corr[edges_select]
 
        # ridge regression
        ridge_grid = GridSearchCV(Ridge(normalize=True),
                               cv=5, param_grid={'alpha': alphas_ridge})
        ridge_grid.fit(x_train, y_train)
        y_pred_ridge = ridge_grid.predict(x_test)
        r2_ridge[k] = r2_score(y_test, y_pred_ridge)
        
        #update the iteration counts
        k=k+1
       
    r2_ridge[r2_ridge < 0] = 0
    r_ridge = np.mean(np.sqrt(r2_ridge))

    return r_ridge


def cpm_clf(all_edges, all_behav, seed, thresh=0.1):
    """Connectome-based Predictive Modeling for classification"""

    # all_behav should be array of size (n,)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    roc_svm = np.zeros(n_splits)
    k = 0
    for train_index, test_index in kf.split(all_edges):
        # split the data into training and test
        x_test, y_test = all_edges[test_index], all_behav[test_index]
        x_train, y_train = all_edges[train_index], all_behav[train_index]
        
        # feature selection
        edges_corr, edges_p = corr_pval(x_train, y_train)
        edges_select = edges_p < thresh 
        x_train = x_train[:, edges_select]
        x_test = x_test[:, edges_select]
        edges_corr = edges_corr[edges_select]
 
        # svc classifier
        svc_clf = LinearSVC()
        svc_clf.fit(x_train, y_train)
        #y_pred = svc_clf.predict(x_test)
        roc_svm[k] = roc_auc_score(y_test, svc_clf.decision_function(x_test))
        
        #update the iteration counts
        k=k+1
       
    roc_svm = np.mean(roc_svm)

    return roc_svm




