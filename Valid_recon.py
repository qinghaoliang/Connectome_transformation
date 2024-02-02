import numpy  as np
import scipy.io as sio
from ot import gromov_wasserstein, unif
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import f_regression, f_classif
from Optimal_transport import load_data_sc, load_data_fc, Optimal_transport
import argparse
from spectral_embedding import spectral_embedding
from scipy.stats import pearsonr
from mats2edges import com2edges, mats2edges
from corr_pval import corr_pval
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser(description='Test mapping on FC of HCP dataset')
parser.add_argument('-s',type=str,help="source atlas")
parser.add_argument('-t',type=str,help="target atlas")
parser.add_argument('-task',type=str,help="task")
args = parser.parse_args()

source = args.s
target = args.t
task = args.task
p = 50

"""load data and mapping"""
fname = "./map/map_fc_" + source + "_" + target + ".npz"
mapping = np.load(fname)
Trans = mapping['Trans']
wm = np.load("hcp_wm.npy")
id = ~np.isnan(wm)
y = wm[id]

comS = load_data_fc(source, task)
comT = load_data_fc(target, task)
comS = comS[:,:,id]
comT = comT[:,:,id]
nsub = np.shape(comS)[2]
nnode = np.shape(comT)[0]
comR = np.zeros((nnode, nnode, nsub))
rot = np.zeros(nsub)
pot = np.zeros(nsub)

"""reconstruction accuracy"""
for i in range(nsub):
    com_s = comS[:,:,i]
    com_t = comT[:,:,i]
    AT = com_t
    AS = com_s
    Xp, Xn = spectral_embedding(AS, p=p)
    Xtp = Trans.T@Xp
    Xtn = Trans.T@Xn
    ATp = Xtp@Xtp.T
    ATn = Xtn@Xtn.T
    Ar = ATp-ATn
    comR[:,:,i] = Ar
    rot[i], pot[i] = pearsonr(com2edges(AT), com2edges(Ar))

print(np.mean(rot))

"""prediction accuracy"""
edgesR = mats2edges(comR)
edgesT = mats2edges(comT)

alphas_ridge = 10**np.linspace(2, -5, 10)

# fit the model on the original data 
X = edgesT
ridge_grid = GridSearchCV(Ridge(), cv=5, param_grid={'alpha': alphas_ridge})
ridge_grid.fit(X, y)
y_pred = ridge_grid.predict(X)
r_iq = []
r_iq.append(pearsonr(y_pred, y))

# evaluate the model on the transformed data
X1 = edgesR
y_pred = ridge_grid.predict(X1)
r_iq.append(pearsonr(y_pred, y))

fname = "./results/recon_fc_" + source + "_" + target + ".npz"
np.savez(fname, rot=rot, pot=pot, r_iq=r_iq)

