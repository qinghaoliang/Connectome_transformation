import numpy  as np
import scipy.io as sio
from ot import gromov_wasserstein, unif
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import f_regression, f_classif
from Optimal_transport import Optimal_transport
from utils import load_data_sc, load_data_fc
from models import cpm_ot, CPM
import argparse
from spectral_embedding import spectral_embedding
from scipy.stats import pearsonr
from mats2edges import com2edges, mats2edges
from corr_pval import corr_pval
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser(description='cv')
parser.add_argument('-s',type=str,help="source atlas")
parser.add_argument('-t',type=str,help="target atlas")
parser.add_argument('-sd',type=int,help="seed")
args = parser.parse_args()

source = args.s
target = args.t
seed = args.sd

## load all y
wm = np.load("hcp_wm.npy")
age = np.load("hcpd_age.npy")

## test the original performance
modality = ["fc","sc"]
results_original = np.zeros((2,2))
at = target
results_original[0, 0], results_original[0, 1] = CPM(at, "fc", "ridge", wm, seed)
results_original[1, 0], results_original[1, 1] = CPM(at, "sc", "ridge", age, seed)

## test the performance using transformed data
modality = ["fc", "sc", "sc_fcmap"]
trans_set = ["train", "test"]
results_ot = np.zeros((4,2,2))
for j in range(len(trans_set)):
    ts = trans_set[j]
    results_ot[0, 0, j],results_ot[0, 1, j] = cpm_ot(source, target, "fc", "ridge", ts, wm, seed)
    results_ot[1, 0, j],results_ot[1, 1, j] = cpm_ot(source, target, "sc", "ridge", ts, age, seed)
    results_ot[2, 0, j],results_ot[2, 1, j] = cpm_ot(source, target, "fc_scmap", "ridge", ts, wm, seed)
    results_ot[3, 0, j],results_ot[3, 1, j] = cpm_ot(source, target, "sc_fcmap", "ridge", ts, age, seed)

fname = "./results/cv_" + source + "_" + target + "_" + str(seed) + ".npz"
np.savez(fname, results_ot=results_ot, results_original=results_original)

