import numpy  as np
import scipy.io as sio
from ot import gromov_wasserstein, unif
from sklearn.preprocessing import normalize
from Optimal_transport import Optimal_transport
from utils import load_data_mapping_time, load_data_fc
import argparse
from spectral_embedding import spectral_embedding
from scipy.stats import pearsonr
from mats2edges import com2edges


"""Learn the mapping across atlases using optimal transport
 paired node-wise time series data is required 
"""

parser = argparse.ArgumentParser(description='learn FC mapping')
parser.add_argument('-s',type=str,help="source atlas")
parser.add_argument('-t',type=str,help="target atlas")
args = parser.parse_args()

source = args.s
target = args.t
task = "wm"
p = 50

dataS = load_data_mapping_time(source)
dataT = load_data_mapping_time(target)

T_time = Optimal_transport(dataS, dataT)
Trans = np.mean(T_time, axis=2)

comS = load_data_fc(source, task)
comT = load_data_fc(target, task)
nsub = np.shape(comS)[2]
rot = np.zeros(nsub)

for i in range(nsub):
    com_s = comS[:,:,i]
    com_t = comT[:,:,i]
    AT = com_t
    AS = com_s
    Xp, Xn = spectral_embedding(AT, p=p)
    ATp = Xp@Xp.T
    ATn = Xn@Xn.T
    A1_T = ATp-ATn
    Xp, Xn = spectral_embedding(AS, p=p)
    Xtp = Trans.T@Xp
    Xtn = Trans.T@Xn
    ATp = Xtp@Xtp.T
    ATn = Xtn@Xtn.T
    A1 = ATp-ATn
    rot[i] = pearsonr(com2edges(A1_T), com2edges(A1))[0]

print(np.mean(rot))

fname = "./map/map_fc_" + source + "_" + target + ".npz"
np.savez(fname, Trans=Trans, rot=rot)

