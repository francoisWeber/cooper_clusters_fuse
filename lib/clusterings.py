# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import math
from sklearn.cluster import DBSCAN, OPTICS

# import our libs
import sys
import os
base_dir = os.path.dirname(__file__) + '/../'
try:
    sys.path.insert(1, base_dir)
except BaseException:
    sys.path.insert(1, os.path.join(base_dir))
from lib import files
from lib import utils


# -------------------
# ----- METRICS -----
# -------------------
def metric_spatial():
    '''
    assuming it will receive (latitude, longitude) like data
    '''
    return lambda x1, x2: utils.geodesic_distance(x1, x2)


def metric_temporal():
    return lambda x1, x2: abs(x1 - x2)


def metric_spatiotemporal(alpha):
    '''
    Function giving a mix-metric for spatio temporal data
    assuming it will receive (latitude, longitude, timestamp) like data
    '''
    return lambda x1, x2: alpha * \
        utils.geodesic_distance(x1[:2], x2[:2]) + \
        (1 - alpha) * abs(x1[2] - x2[2])


# ------------------------------------
# ----- OPTICS LABELS EXTRACTION -----
# ------------------------------------
def optics_extract_sklearn_labels():
    return lambda clust: clust.labels_


def optics_extract_dbscan_labels(epsilon):
    return lambda clust: clust.extract_dbscan(epsilon)[1]


# ------------------------------
# ----- CLUSTERING METHODS -----
# ------------------------------
def optics(data, metric, label_extraction,
           min_samples=1, rejection_ratio=0.1, maxima_ratio=0.25):
    # instanciate
    clust = OPTICS(metric=metric,
                   min_samples=min_samples,
                   rejection_ratio=rejection_ratio,
                   maxima_ratio=maxima_ratio)
    # fit
    clust.fit(data)
    # extract clustering
    labels = label_extraction(clust)
    return labels, clust


def dbscan(data, metric, epsilon, min_samples=1):
    # instanciate
    clust = DBSCAN(metric=metric,
                   min_samples=min_samples,
                   eps=epsilon)
    # fit
    clust.fit(data)
    # extract clustering
    labels = clust.labels_
    return labels, clust
