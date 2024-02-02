import numpy  as np
import scipy.io as sio
from ot import gromov_wasserstein, unif
from sklearn.preprocessing import normalize
from utils import load_data_mapping_sc, load_data_sc
import argparse
from spectral_embedding import spectral_embedding
from scipy.stats import pearsonr
from mats2edges import com2edges

"""Learn the mapping across atlases using graph matching 
paired connectivity data is required
"""

parser = argparse.ArgumentParser(description='learn SC mapping')
parser.add_argument('-s',type=str,help="source atlas")
parser.add_argument('-t',type=str,help="target atlas")
args = parser.parse_args()

source = args.s
target = args.t
task = "qa_pass"
p = 50

dataS = load_data_mapping_sc(source, task)
dataT = load_data_mapping_sc(target, task)

nsub = np.shape(dataS)[2]
dim_s = np.shape(dataS)[1]
dim_t = np.shape(dataT)[1]
p_s = unif(dim_s)
p_t = unif(dim_t)
T_gw = np.zeros((dim_s, dim_t, nsub))
rgw = np.zeros(nsub)

for i in range(nsub):
    com_s = dataS[:,:,i]
    com_t = dataT[:,:,i]
    trans = gromov_wasserstein(com_s, com_t, p_s, p_t)
    T_gw[:,:,i] = trans

Trans = np.mean(T_gw, axis=2)

dataS = load_data_sc(source, task)
dataT = load_data_sc(target, task)
dim_s = np.shape(dataS)[1]
dim_t = np.shape(dataT)[1]
nsub = np.shape(dataS)[2]
for i in range(nsub):
    com_s = dataS[:,:,i]
    com_t = dataT[:,:,i]
    AT = com_t
    AS = com_s
    Xp, Xn = spectral_embedding(AT, p=p)
    ATp = Xp@Xp.T
    ATn = Xn@Xn.T
    A1_T = ATp-ATn
    Xp, Xn = spectral_embedding(AS, p=p)
    Xtp = dim_t*Trans.T@Xp
    Xtn = dim_t*Trans.T@Xn
    ATp = Xtp@Xtp.T
    ATn = Xtn@Xtn.T
    A1 = ATp-ATn
    rgw[i] = pearsonr(com2edges(A1_T), com2edges(A1))[0]

print(np.mean(rgw))

fname = "./map/map_sc_" + source + "_" + target + ".npz"
np.savez(fname, Trans=Trans, rgw=rgw)

