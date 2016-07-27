from herdingspikes import spikeclass, ImportInterpolated
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import SparsePCA

# load an unsorted data set recorded at 24kHz (1024 channels, 42um pitch)
path = 'data/'
fname = 'P29_16_05_14_retina02_left_stim2_smallarray_fullfield_v28.hdf5'
O = ImportInterpolated(path+fname)
offset_list = np.arange(33, 40, 1)
upto_list = np.arange(63, 70, 1)
# sparsePCA_alpha = [1, 3, 7, 20, 55, 150, 400]
# sparsePCA_ridge_alpha = [0.0001, 0.01, 0.1, 1, 10]

n = 1
sparsePCA_alpha_ind = 20
for FrameRange_ind in range(len(offset_list)):
# for sparsePCA_alpha_ind in sparsePCA_alpha:
    # for sparsePCA_ridge_alpha_ind in sparsePCA_ridge_alpha:
    # compute PCA
    ncomp = 5
    offset = offset_list[FrameRange_ind]
    upto = upto_list[FrameRange_ind]
    # if ~upto:
    #     upto = O.Shapes().shape[0]
    PCA_start = time.time()
    p = SparsePCA(n_components=ncomp, alpha=sparsePCA_alpha_ind, ridge_alpha=0.01)
    PCA_end = time.time()
    print("The " + str(n) + " PCA time: " + str(PCA_end-PCA_start))
    Projection_start = time.time()
    scorePCA = p.fit_transform(O.Shapes()[offset:upto, :].T).T
    Projection_end = time.time()
    print("The " + str(n) + " Projection time: " + str(Projection_end-Projection_start))
    # explained_variance_ratio = p.explained_variance_ratio_
    plt.figure(1)
    plt.plot(p.components_.T)
    plt.legend(range(5))
    plt.savefig("princomp/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) + ".png", bbox_inches='tight')
    plt.clf()

    plt.figure(2)
    plt.scatter(scorePCA[0, :10000], scorePCA[1, :10000], s=4)
    plt.savefig("scatter/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) + ".png", bbox_inches='tight')
    plt.clf()
    n = n+1
