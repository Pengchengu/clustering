from herdingspikes import spikeclass, ImportInterpolated
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import FastICA

# load an unsorted data set recorded at 24kHz (1024 channels, 42um pitch)
path = 'data/'
fname = 'P29_16_05_14_retina02_left_stim2_smallarray_fullfield_v28.hdf5'
O = ImportInterpolated(path+fname)

# offset_list = np.arange(14, 26, 2)
# upto_list = np.arange(58, 70, 2)
# MeanShift_alpha = [0.28, 2]
CompNumb = [40, 44]
# CompNumb_meanshift = [2, 4, 8, 16, 24, 32, 40, 44]

fp_list = []
fn_list = []
f1_list = []


n = 1
# for FrameRange_ind in range(len(offset_list)):
for CompNumb_ind in CompNumb:
    # compute PCA
    ncomp = CompNumb_ind
    offset = 18
    upto = 62
    # ncomp = 44
    # offset = offset_list[FrameRange_ind]
    # upto = upto_list[FrameRange_ind]
    # if ~upto:
    #     upto = O.Shapes().shape[0]
    ICA_start = time.time()
    p = FastICA(n_components=ncomp)
    ICA_end = time.time()
    print("The " + str(n) + " ICA time: " + str(ICA_end-ICA_start))
    Projection_start = time.time()
    scorePCA = p.fit_transform(O.Shapes()[offset:upto, :].T).T
    Projection_end = time.time()
    print("The " + str(n) + " Projection time: " + str(Projection_end-Projection_start))
    # explained_variance_ratio = p.explained_variance_ratio_
    plt.figure(1)
    plt.plot(p.components_.T)
    plt.legend(range(5))
    plt.savefig("princomp/" + str(offset) + "to" + str(upto) + "CompNumb" + str(ncomp) + ".pdf", bbox_inches='tight')
    plt.clf()

    plt.figure(2)
    plt.scatter(scorePCA[0, :10000], scorePCA[1, :10000], s=4)
    plt.savefig("scatter/" + str(offset) + "to" + str(upto) + "CompNumb" + str(ncomp) + ".pdf", bbox_inches='tight')
    plt.clf()

    m = 1
    CompNumb_meanshift = [2, ncomp]
    MeanShift_alpha_ind = 2
    for CompNumb_meanshift_ind in CompNumb_meanshift:
        # for MeanShift_alpha_ind in MeanShift_alpha:
        # Cluster the remaining events into single units
        MeanShift_start = time.time()
        # O.CombinedMeanShift(0.3, MeanShift_alpha_ind, PrincComp=scorePCA, mbf=10, njobs=1)
        O.CombinedMeanShift(0.3, MeanShift_alpha_ind, PrincComp=scorePCA[:CompNumb_meanshift_ind, :], mbf=10, njobs=1)

        MeanShift_end = time.time()
        print("The " + str(m) + " MeanShift time: " + str(MeanShift_end-MeanShift_start))
        print('Found '+str(O.NClusters())+' clusters for '+str(O.NData())+' spikes.')
        # show waveforms of 12 example clusters
        # O.ShapesPlot()
        plt.figure(3)
        O.PlotRegion((35, 40, 50, 55), ax=plt.gca())
        plt.savefig("clusters/" + str(offset) + "to" + str(upto) + "_CompNumb" + str(CompNumb_meanshift_ind) +
                    "of" + str(ncomp) + "_meanshift" + str(MeanShift_alpha_ind) + ".pdf", bbox_inches='tight')
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
            cl = np.hstack((c, Q.Neighbours(c, 1., min_neigh_size=100)))
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
        plt.savefig("fn_fp/" + str(offset) + "to" + str(upto) + "_CompNumber" + str(CompNumb_meanshift_ind) +
                    "of" + str(ncomp) + "_meanshift" + str(MeanShift_alpha_ind) + ".pdf", bbox_inches='tight')
        # plt.savefig("fn_fp/" + str(offset) + "to" + str(upto) +
        #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
        plt.clf()

        plt.figure(5)
        plt.hist(f1, bins=np.arange(0, 1.05, 0.05), weights=np.zeros_like(f1) + 1. / f1.size)
        plt.title('F1 score')
        plt.ylabel('Number of units')
        plt.savefig("f1/" + str(offset) + "to" + str(upto) + "_CompNumber" + str(CompNumb_meanshift_ind) +
                    "of" + str(ncomp) + "_meanshift" + str(MeanShift_alpha_ind) + ".pdf", bbox_inches='tight')
        # plt.savefig("f1/" + str(offset) + "to" + str(upto) +
        #             "_meanshift" + str(MeanShift_alpha_ind) + ".png", bbox_inches='tight')
        plt.clf()


        fp_list.append(np.nan_to_num(fp))
        fn_list.append(np.nan_to_num(fn))
        f1_list.append(np.nan_to_num(f1))
        m = m + 1
    n = n + 1


print("false postives", fp_list)
print("false negatives", fn_list)
print("f1 measure", f1_list)
plt.figure(6)
plt.title('False positives Trend')
plt.xlabel('Times')
plt.boxplot(fp_list)
plt.savefig("fn_fp/False positives Trend.pdf", bbox_inches='tight')
plt.clf()

plt.figure(7)
plt.title('False negatives Trend')
plt.boxplot(fn_list)
plt.savefig("fn_fp/False negatives Trend.pdf", bbox_inches='tight')
plt.clf()

plt.figure(8)
plt.title('F-measure Trend')
plt.xlabel('Times')
plt.boxplot(f1_list)
plt.savefig("f1/F-measure Trend.pdf", bbox_inches='tight')
plt.clf()

# plt.show()














