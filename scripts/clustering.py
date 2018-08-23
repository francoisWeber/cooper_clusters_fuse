# -*- coding: utf-8 -*-
import os
import sys
import argparse

base_dir = os.path.dirname(__file__) + '/../'
try:
    sys.path.insert(1, base_dir)
except BaseException:
    sys.path.insert(1, os.path.join(base_dir))
from lib import clusterings as clust
from lib import files 
from lib import utils

import numpy as np

# prepare parser
parser = argparse.ArgumentParser(description='help to choose a clustering')
# clustering method
parser.add_argument('-a',
                    choices=['dbscan', 'optics'],
                    dest='algo',
                    metavar='algorithm to use', 
                    type=str, 
                    default='dbscan',
                    help='a clustering method : "dbscan" or "optics"')
# data to use
parser.add_argument('-f',
                    choices=['spatial', 'temporal', 'spatio-temporal'],
                    dest='features',
                    metavar='features to use', 
                    type=str, 
                    default='spatio-temporal',
                    help='a kind of features : "spatial" or "temporal" or "spatio-temporal" (default)')
# directory containing photos
parser.add_argument('-d',
                    dest='dir_path',
                    metavar='directory containing photos', 
                    type=str, 
                    help='relative path to the photo directory')
# epsilon value to be used with DBSCAN
parser.add_argument('-e',
                    dest='eps',
                    metavar='epsilon value for DBSCAN', 
                    type=float, 
                    default=1.0,
                    help='epsilon scalar to be used by DBSCAN')
# alpha value ruling the mix between spatial and temporal distances 
parser.add_argument('--alpha',
                    dest='alpha',
                    metavar='alpha value mixing distances (0=pure temporal ; 1=pure geodesic)', 
                    type=float, 
                    default=0.5,
                    help='alpha scalar used to mix distances')
# if OPTICS : what kind of label extraction ? 
parser.add_argument('--label-extraction',
                    choices=['sklearn', 'dbscan'],
                    dest='label_extraction',
                    metavar='method used to extract labels from OPTICS', 
                    type=str, 
                    default='sklearn',
                    help='method used to extract labels from OPTICS : "sklearn" or "dbscan"')
args = parser.parse_args()

# parse arguments
algo = args.algo
features = args.features
eps = args.eps
dir_path = args.dir_path
alpha = args.alpha
label_extraction = args.label_extraction

# load relevant data
if features == 'temporal':
    res = files.scan_images(dir_path, False)
    data = np.array(res['timestamps']).reshape(-1, 1) / 3600
    metric = clust.metric_temporal()
elif features == 'spatial':
    res = files.scan_images(dir_path, True)
    coords = res['coords']
    latitudes = np.array([coord['lat'] for coord in coords]).reshape(-1, 1)
    longitudes = np.array([coord['long'] for coord in coords]).reshape(-1, 1)
    data = np.concatenate((latitudes, longitudes), axis=1)
    metric = clust.metric_spatial()
elif features == 'spatio-temporal':
    res = files.scan_images(dir_path, True)
    coords = res['coords']
    latitudes = np.array([coord['lat'] for coord in coords]).reshape(-1, 1)
    longitudes = np.array([coord['long'] for coord in coords]).reshape(-1, 1)
    timestamps = np.array(res['timestamps']).reshape(-1, 1) / 3600
    data = np.concatenate((latitudes, longitudes, timestamps), axis=1)
    metric = clust.metric_spatiotemporal(alpha)

# if OPTICS : select the label extractor
if label_extraction == 'sklearn':
    label_extraction = clust.optics_extract_sklearn_labels()
elif label_extraction == 'dbscan':
    label_extraction = clust.optics_extract_dbscan_labels(alpha)

# algorithm 
print('Clustering ...')
if algo == 'dbscan':
    labels, _ = clust.dbscan(data, metric, eps)
elif algo == 'optics':
    labels, _ = clust.optics(data, metric, label_extraction)


# save results
files.save_dbscan_results(labels, res['filenames'])
utils.start_server(dir_path)
