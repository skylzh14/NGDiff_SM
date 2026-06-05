import hdf5storage as hdf5
import numpy as np

data = np.load("../data/t5_0_full.pkl.npy")
print(",,,,,,,,")
hdf5.savemat("../512_512_feature_gamma.mat",{'feature':data})
