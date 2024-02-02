import numpy as np
from sklearn.utils.extmath import row_norms
from scipy import stats

def corr_pval(X_in, y):
# compute the correlation and pvalue for (X, y) with missing entries
# all missing entries have value nan
    X = X_in.copy()
    n_features = X.shape[1]
    Y = np.tile(y, (n_features, 1))
    Y = Y.T
    mis = np.isnan(X)
    Y[mis] = np.nan
    X[mis] = np.nan
    Y_means = np.nanmean(Y, axis=0)
    X_means = np.nanmean(X, axis=0)
    Y = Y - Y_means
    Y[mis] = 0
    X[mis] = 0
    
    n_sample = np.sum(1-mis.astype(int), axis=0)
    X_norms = np.sqrt(row_norms(X.T, squared=True) -
                          n_sample * X_means ** 2)
    X_norms[X_norms==0] = np.nan
    # compute the correlation
    corr = np.sum(np.multiply(Y, X), axis=0)
    corr /= X_norms
    y_sd = row_norms(Y.T)
    corr /= y_sd

    # convert to p-value
    degrees_of_freedom = n_sample - 2
    F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    pv = stats.f.sf(F, 1, degrees_of_freedom)
    return corr, pv


    
