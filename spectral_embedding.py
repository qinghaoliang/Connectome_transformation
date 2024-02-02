import numpy as np
from sklearn.manifold import Isomap

"""Embedding the connectomes to the latent space"""

def spectral_embedding(A, p=50):
    ## spectral embedding
    # the outputs are the positive and negative spectrum
    w, v = np.linalg.eigh(A)
    ind = (np.abs(w)).argsort()[::-1][:p]
    wp = w[ind[w[ind]>0]]
    vp = v[:, ind[w[ind]>0]]
    wn = w[ind[w[ind]<0]]
    vn = v[:, ind[w[ind]<0]]
    Xp = vp@np.diag(np.sqrt(np.abs(wp)))
    Xn = vn@np.diag(np.sqrt(np.abs(wn)))
    return Xp, Xn

def spectral_embedding_isomap(A, p=50, d=10):
    ## spectral embedding with additional nonlinear dimenson reduction
    w, v = np.linalg.eigh(A)
    ind = (np.abs(w)).argsort()[::-1][:p]
    wp = w[ind[w[ind]>0]]
    vp = v[:, ind[w[ind]>0]]
    wn = w[ind[w[ind]<0]]
    vn = v[:, ind[w[ind]<0]]
    Xp = vp@np.diag(np.sqrt(np.abs(wp)))
    Xn = vn@np.diag(np.sqrt(np.abs(wn)))
    
    embed_p = Isomap(n_components=d)
    Zp = embed_p.fit_transform(Xp)
    embed_n = Isomap(n_components=d)
    Zn = embed_n.fit_transform(Xn)

    return Zp, Zn

def spectemb(A, p=50):
    ## normal matrix factorization
    w, v = np.linalg.eigh(A)
    ind = (np.abs(w)).argsort()[::-1][:p]
    wp = w[ind]
    vp = v[:, ind]
    ind = wp.argsort()[::-1]
    wp = wp[ind]
    vp = vp[:, ind]
    X = vp@np.diag(np.sqrt(np.abs(wp)))
    return X