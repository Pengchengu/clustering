from herdingspikes import spikeclass, ImportInterpolated
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA
from matplotlib.backends.backend_pdf import PdfPages

# load an unsorted data set recorded at 24kHz (1024 channels, 42um pitch)
path = 'data/'
fname = 'P29_16_05_14_retina02_left_stim2_smallarray_fullfield_v28.hdf5'
O = ImportInterpolated(path+fname)
# plt.figure(figsize = (9, 4))
# ax = plt.subplot(121)


# O.LogHistPlot(ax=ax, binstep=0.1)
# ax = plt.subplot(122)
# O.DataPlot(ax=ax, show_max=int(2e4))

# offset_list = np.arange(20, 40, 2)
# upto_list = np.arange(50, 70, 2)
# sparsePCA_alpha = [1, 20, 50, 100, 150, 200]
# sparsePCA_ridge_alpha = [0.0001, 0.01, 0.1, 1, 10]
# MeanShift_alpha = [10, 30, 50, 70, 90]
CompNumb = [2, 3, 4, 5]

fp_mean = []
fn_mean = []

n = 1
# for FrameRange_ind in range(len(offset_list)):
# for sparsePCA_alpha_ind in sparsePCA_alpha:
# for sparsePCA_ridge_alpha_ind in sparsePCA_ridge_alpha:
# compute PCA
ncomp = 5
sparsePCA_alpha_ind = 50
# offset = offset_list[FrameRange_ind]
# upto = upto_list[FrameRange_ind]
offset = 28
upto = 58
# if ~upto:
#     upto = O.Shapes().shape[0]
# p = PCA(n_components=ncomp, whiten=True)
PCA_start = time.time()
# p = FastICA(n_components=ncomp)
# p = PCA(n_components=ncomp, whiten=True)
p = SparsePCA(n_components=ncomp, alpha=sparsePCA_alpha_ind, ridge_alpha=0.01)
PCA_end = time.time()
print("The " + str(n) + " PCA time: " + str(PCA_end-PCA_start))
# p = FastICA(n_components=ncomp)
Projection_start = time.time()
scorePCA = p.fit_transform(O.Shapes()[offset:upto, :].T).T
Projection_end = time.time()
print("The " + str(n) + " Projection time: " + str(Projection_end-Projection_start))
# explained_variance_ratio = p.explained_variance_ratio_
plt.figure(1)
plt.plot(p.components_.T)
plt.legend(range(5))
plt.savefig("princomp/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) + ".png", bbox_inches='tight')
# plt.savefig("princomp/" + str(offset) + "to" + str(upto) + ".png", bbox_inches='tight')
plt.clf()

plt.figure(2)
plt.scatter(scorePCA[0, :10000], scorePCA[1, :10000], s=4)
plt.savefig("scatter/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) + ".png", bbox_inches='tight')
# plt.savefig("scatter/" + str(offset) + "to" + str(upto) + ".png", bbox_inches='tight')
plt.clf()

m = 1
MeanShift_alpha_ind = 50
for CompNumber in CompNumb:
    # for MeanShift_alpha_ind in MeanShift_alpha:
    # Cluster the remaining events into single units
    MeanShift_start = time.time()
    # O.CombinedMeanShift(0.3, MeanShift_alpha_ind, PrincComp=scorePCA[:2, :], mbf=10, njobs=1)
    O.CombinedMeanShift(0.3, MeanShift_alpha_ind, PrincComp=scorePCA[:CompNumber, :], mbf=10, njobs=1)

    MeanShift_end = time.time()
    print("The " + str(m) + " MeanShift time: " + str(MeanShift_end-MeanShift_start))
    print('Found '+str(O.NClusters())+' clusters for '+str(O.NData())+' spikes.')
    # show waveforms of 12 example clusters
    # O.ShapesPlot()
    plt.figure(3)
    O.PlotRegion((35, 40, 50, 55), ax=plt.gca())
    # plt.savefig("clusters/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.savefig("clusters/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
                "_CompNumber" + str(CompNumber) + ".png", bbox_inches='tight')
    # plt.savefig("clusters/" + str(offset) + "to" + str(upto) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.clf()

    # Assessing the quality of the clustering by computing a confusion matrix
    # using a fit with a mixtures of Gaussians model
    Q = O.QualityMeasures(scorePCA=scorePCA)
    # Here all clusters with at least 100 spikes are assessed in turn.
    # Their nearest neighbours are found, and a Gaussian mixture is fit to the combined data.
    # Responsibilities are then used to compute a confusion maric, which yields false positives and negatives.
    clusters = np.where(O.ClusterSizes() > 100)[0]
    fp = np.zeros(len(clusters))
    fn = np.zeros(len(clusters))
    for i, c in enumerate(clusters):
        cl = np.hstack((c,Q.Neighbours(c, 1., min_neigh_size=100)))
        conf = Q.GaussianOverlapGroup(cl)
        # conf2 = Q.GaussianOverlapGroup(np.append(cl, nns), mode='PCA')
        fp[i] = conf[0, 0]
        fn[i] = np.sum(conf-np.diag(np.diag(conf)), axis=0)[0]

    f1 = 2*(1-fp)/(2*(1-fp)+fp+fn)
    plt.figure(4)
    # plt.figure(figsize=(10, 3.25))
    plt.subplot(121)
    plt.hist(fp, bins=np.arange(0, 1, 0.05), weights=np.zeros_like(fp) + 1. / fp.size)
    plt.title('False positives')
    plt.ylabel('Number of units')
    plt.subplot(122)
    plt.hist(fn, bins=np.arange(0, 1, 0.05), weights=np.zeros_like(fn) + 1. / fn.size)
    plt.title('False negatives')
    # plt.savefig("fn_fp/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.savefig("fn_fp/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
                "_CompNumber" + str(CompNumber) + ".png", bbox_inches='tight')
    # plt.savefig("fn_fp/" + str(offset) + "to" + str(upto) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.clf()

    plt.figure(5)
    plt.hist(f1, bins=np.arange(0, 1.05, 0.05), weights=np.zeros_like(f1) + 1. / f1.size)
    plt.title('F1 score')
    plt.ylabel('Number of units')
    # plt.savefig("f1/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.savefig("f1/" + str(offset) + "to" + str(upto) + "_alpha" + str(sparsePCA_alpha_ind) +
                "_CompNumber" + str(CompNumber) + ".png", bbox_inches='tight')
    # plt.savefig("f1/" + str(offset) + "to" + str(upto) +
    #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
    plt.clf()

    fp_mean.append(fp.mean())
    fn_mean.append(fn.mean())
    m = m + 1
n = n + 1

plt.figure(6)
plt.subplot(121)
plt.plot(fp_mean)
print()
plt.title('False positives Trend')
plt.ylabel('Average values ')
plt.xlabel('Times')
plt.subplot(122)
plt.plot(fn_mean)
plt.title('False negatives Trend')
plt.ylabel('Average values')
plt.xlabel('Times')
plt.savefig("fn_fp/False positives and False negatives Trend.png", bbox_inches='tight')
plt.clf()
# plt.show()












