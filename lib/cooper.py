# -*- coding: utf-8 -*-
# Maths and arrays
import numpy as np
from scipy.spatial import distance_matrix
from scipy.ndimage.filters import convolve
from scipy import stats as st
from scipy import kron
from scipy.signal import find_peaks_cwt
import math

# ML :
from sklearn.cluster import KMeans


def similarity_matrix(feature, bandwidth, metric=None):
    '''
    Compute a similarity matrix for given features wrt given metric
    The similarity matrix is a kernelized distance matrix (using a gaussian
    kernel). If m: (x,y) --> m(x, y) is a metric, then
    S[i, j] = exp(- m(X_i, X_j) / bandwidth)

    INPUTS:
    - feature : feature matrix of shape (n, d) representing n observations of
        a d-dimensional feature vectore
    - bandwidth : a scalar representing the bandwidth of the gaussian kernel
    - metric : optional if one wants the euclidian distance, a callable that
        representing a mathematical metric otherwise

    OUTPUTS:
    - a (n, n) np.ndarray representing the similarity matrix
    '''
    if metric is None:
        distance_t = distance_matrix(feature, feature, 2)
    else:
        N = feature.shape[0]
        distance_t = np.zeros(shape=(N, N))
        for i in range(N):
            for j in range(i + 1, N):
                distance_t[i, j] = metric(feature[i, :], feature[j, :])
        distance_t = distance_t + distance_t.T

    return np.exp(- distance_t / bandwidth)


def gaussian_checkerboard_kernel(L, sigma=10):
    '''
    Computes a gaussian checkerboard kernel of shape (2L+1, 2L+1)
    The result is the Hadamar (element-wise) product of a centered 2D gaussian
    pdf of std `sigma` evaluated on (-L:L)x(-L:L) with a checkerboard kernel.
    Thus the resulting matrix is split into 4 quarts : 2 negatives quarts and
    2 positive parts.

    INPUTS:
    - L : half length of the kernel
    - sigma : optional parameter : the std of the gaussian part of the kernel

    OUTPUS:
    - np.ndarray representing a gaussian checkerboard kernel of shape (2L+1)
    '''
    # get gaussian part
    x, y = np.mgrid[-L:L, -L:L]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = st.multivariate_normal([0, 0], sigma * np.eye(2))
    pdf = rv.pdf(pos)
    # get checkerboard part
    checkerboard = kron(np.array([[1, -1], [-1, 1]]), np.ones(shape=(L, L)))
    # return
    return checkerboard * pdf


def novelty_score(S, L=5, sigma=1):
    '''
    Computing the novelty score based on a similarity matrix as shown in [1]
    The aim of that novelty score is to determine cuts wrt the metric used to
    build the similarity matrix and thus, build clusters. This function is
    described in [1] at equation (3)

    [1] COOPER, Matthew, FOOTE, Jonathan, GIRGENSOHN, Andreas, et al. Temporal
    event clustering for digital photo collections. ACM Transactions on
    Multimedia Computing, Communications, and Applications (TOMM), 2005,
    vol. 1, no 3, p. 269-288.

    INPUTS:
    - S : (n, n) similarity matrix on which to perform the novelty extraction
    - L : optional scalar used as length of the convolution of S by a gaussian
        checkerboard kernel
    - sigma : optional scalar ; the std of the gaussian part of the kernel

    OUTPUTS :
    - vector of dimension n representing the novelty score at each point
    '''

    kernel = gaussian_checkerboard_kernel(L, sigma)
    conv = convolve(S, kernel)
    return np.diag(conv)


def detect_peaks_on_novelty(nu, **kwargs):
    '''
    Function detecting the indices of peaks in novelty score
    Uses a SciPy built-in function find_peaks_cwt to detect a first bundle of
    peaks in the novelty score. This set is then refined to get only the
    highest peaks thanks to a 3-cluster k-means analysis aiming at separating
    low peaks from medium peaks from high peaks.

    INPUTS:
    - nu : np.array of novelty score as given by function novelty_score
    - kwargs : optionnal keyword arguments to give specific arguments to
        find_peaks_cwt or KMeans

    OUTPUTS:
    - indices of peaks in nu
    '''
    # acquire kwargs :
    find_peaks_cwt_width = kwargs.get('width', np.arange(1, 2))
    k_means_n_clusters = kwargs.get('n_clusters', 3)

    # compute gradient of nu
    delta_nu = np.diff(nu)

    # raw peak detection on delta_nu
    p = find_peaks_cwt(delta_nu, find_peaks_cwt_width)

    # casting back peaks to nu indices
    peaks_location = p + 1
    peaks_height = nu[peaks_location]

    # refining : clustering to find higher peaks
    km = KMeans(n_clusters=k_means_n_clusters)
    peaks_clusters = km.fit_predict(peaks_height.reshape(-1, 1))
    cluster_of_interest = km.cluster_centers_.argmax()

    return peaks_location[peaks_clusters == cluster_of_interest]


def compute_boundaries(nu, **kwargs):
    '''
    Cast a list of novelty peaks indices into a list of clusters

    INPUTS:
    - nu : the novelty score on which to base the clustering

    OUTPUS:
    - list of list of photo indices belonging to the same cluster
    '''
    # init
    clusters = []
    N = nu.shape[0]

    # get peaks of the novelty score
    peaks_list = detect_peaks_on_novelty(nu, **kwargs)

    # building lists of clusters
    prev_cut = 0
    for cut in peaks_list:
        clusters.append(np.arange(prev_cut, cut + 1))
        prev_cut = cut + 1
    # adding the last segment
    clusters.append(np.arange(prev_cut, N))

    return clusters


def compute_confidence_score(clustering, S):
    '''
    Compute the confidence interval of a clustering
    This implements the equation (8) of [1]

    [1] COOPER, Matthew, FOOTE, Jonathan, GIRGENSOHN, Andreas, et al. Temporal
    event clustering for digital photo collections. ACM Transactions on
    Multimedia Computing, Communications, and Applications (TOMM), 2005,
    vol. 1, no 3, p. 269-288.

    INPUTS:
    - clustering : a list of list of photo indice representing the clusters.
        Should be as the output of compute_boundaries()

    OUTPUTS:
    - score : confidence score of the given cluster
    '''
    score = 0
    for (i, c) in enumerate(clustering):
        # within-cluster mean similarity
        score += S[np.ix_(c, c)].mean()
        # between-clusters mean similarity
        if i < len(clustering) - 1:
            c_next = clustering[i + 1]
            score -= S[np.ix_(c, c_next)].mean()

    return score


def compute_clusterings_at_scales(feature, bandwidths, metric=None):
    clusterings_infos = []
    for (i, bw) in enumerate(bandwidths):
        S_bw = similarity_matrix(feature, bw, metric)
        nu_bw = novelty_score(S_bw)
        clustering_bw = compute_boundaries(nu_bw)
        score_i = compute_confidence_score(clustering_bw, S_bw)
        clusterings_infos.append({
            'bandwidth': bw,
            'similarity_matrix': S_bw,
            'novelty_score': nu_bw,
            'clusters_indices': clustering_bw,
            'score': score_i
        })
    return clusterings_infos


def compute_clustering(feature, bandwidths, metric=None):
    '''
    Select the best cluster wrt to confidence among many scales of clusters
    Computes a clustering for every given bandwidth and select the one having
    the highest confidence score. Method described in algo 3 of [1]

    [1] COOPER, Matthew, FOOTE, Jonathan, GIRGENSOHN, Andreas, et al. Temporal
    event clustering for digital photo collections. ACM Transactions on
    Multimedia Computing, Communications, and Applications (TOMM), 2005,
    vol. 1, no 3, p. 269-288.

    INPUTS
    - features : features on which to base the clustering
    - bandwidhts : a list of bandwidths to use that describes the different
        scales of the feature
    - metric : optional metric to use on the features

    OUTPUTS :
    - best_clustering : the best clustering wrt confidence score
    '''
    score_max = 0
    best_clustering = None
    for (i, bw) in enumerate(bandwidths):
        S_bw = similarity_matrix(feature, bw, metric=None)
        nu_bw = novelty_score(S_bw)
        clustering_bw = compute_boundaries(nu_bw)
        score_i = compute_confidence_score(clustering_bw, S_bw)
        if score_max < score_i:
            score_max = score_i
            best_clustering = clustering_bw
    return best_clustering

# Part relative to second Cooper's paper :


def clusters_probability(R, N=None):
    card_r = [r.size for r in R]
    if N is None:
        N = sum(card_r)
    return [c / N for c in card_r]


def clustering_entropy(R, N):
    P = clusters_probability(R, N)
    H = - sum([p * math.log(p) for p in P])
    return H


def clusters_conditional_info(R, s, N):
    set_s = set(s)
    I_R_knowing_s = 0
    for r in R:
        card_r_and_s = len(set(r).intersection(set_s))
        if card_r_and_s > 0:
            P_r_knowing_s = card_r_and_s / len(s)
            P_r = len(r) / N
            I_R_knowing_s += P_r_knowing_s * math.log(P_r_knowing_s / P_r)
    return I_R_knowing_s


def nmi_criterion_between_indices(cal_R, b_i, b_j, N):
    '''
    Computes the E_{NMI} score expressed at equation (13) in [1]
    '''
    # the potential S-cluster to test is thus
    s = np.arange(b_i, b_j)
    P_s = (b_j - b_i) / N
    criterion = 0
    for R in cal_R:
        criterion += clusters_conditional_info(R,
                                               s, N) / clustering_entropy(R, N)
    criterion = criterion * P_s / len(cal_R)
    return criterion


def set_of_all_boundaries(cal_R):
    all_boundaries = []
    for R in cal_R:
        # don't incude the first photo because we won't use it
        for r in R:
            all_boundaries.append(r[0])
    # include the last boundary == N
    all_boundaries.append(sum([r.size for r in R]))
    all_boundaries = np.unique(np.array(all_boundaries))
    return all_boundaries


def score_for_clustering(cal_R, N):
    '''
    Apply equation (7) to find optimal clustering

    NOT SURE IT WORKS : WEIRD RESULTS

    EXAMPLE:
    # compute 2 sets ofs clusterings
    feature_temporal = timestamp_ordered.reshape(-1, 1)
    clusterings_temporal, info_temp = compute_clusterings_at_scales(feature_temporal, np.arange(1, 36)*60*60)
    feature_spatial = data.get_values()
    clusterings_spatial, info_sp = compute_clusterings_at_scales(feature_spatial, np.arange(1, 10), geodesic_distance)
    # set of possible clusterings
    cal_R = clusterings_spatial + clusterings_temporal
    # apply selection
    e, b = score_for_clustering(cal_R, feature_temporal.shape[0])
    '''
    # get all possible boundaries
    Beta = set_of_all_boundaries(cal_R)

    # now increment on k to build successively the scores of every possibles clusterings
    # prepare the storage of all possible values of E_nmi(b_j, k)
    E_nmi = np.zeros(shape=(len(Beta), len(Beta)))
    boundary = np.zeros(shape=(len(Beta), len(Beta) - 3), dtype=int)
    for k in range(2, len(Beta) - 2):
        for (j, b_j) in enumerate(Beta):
            if b_j >= k:
                reachable_i = (k <= Beta) * (Beta <= b_j) * \
                    [b_i not in boundary[j, :] for b_i in Beta]
                if reachable_i.any():
                    # corresponds "max_{b_i \in Beta / k<=b_i<=b_j}"
                    reachable_b_i = Beta[reachable_i]
                    C_nmi = [
                        nmi_criterion_between_indices(
                            cal_R, b_i, b_j, N) for b_i in reachable_b_i]
                    relevant_scores = C_nmi + E_nmi[reachable_i, k - 1]
                    i_star = np.argmax(relevant_scores)
                    E_nmi[j, k] = relevant_scores[i_star]
                    # now get the index that maximizes the E_NMI[:, k]
                    boundary[j, k - 2] = reachable_b_i[i_star]
    boundary = boundary[:, 2:]
    return E_nmi, boundary
