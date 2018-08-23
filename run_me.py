
# coding: utf-8
import json

import numpy as np

from lib import cooper
from lib import clusterings as clust
from lib import files


# ## Load data ...
# load the metadata of images in Dataset : timestamp and GPS
try:
    with open('./res.json', 'r') as f:
        res = json.load(f)
except BaseException:
    res = files.scan_images('Photos/fweber.mycozy.cloud/', True)
    with open('./res.json', 'w') as f:
        json.dump(res, f, indent=4)
N = len(res['filenames'])


# ## clustering based on spatial data, then temporal data
# ### Spatial ...
# extract spatial data
coords = res['coords']
latitudes = np.array([coord['lat'] for coord in coords]).reshape(-1, 1)
longitudes = np.array([coord['long'] for coord in coords]).reshape(-1, 1)
data = np.concatenate((latitudes, longitudes), axis=1)
metric = clust.metric_spatial()
# build a first set of spatial clusterings
# specify bandwidth (in km) to be used in the spatial kernel :
bandwidths = [1, 2, 4, 8, 16, 32]
R_spatial = cooper.compute_clusterings_at_scales(
    feature=data, bandwidths=bandwidths, metric=metric)
# extract the boundaries of every clustering
clust_spatial = [r['clusters_indices'] for r in R_spatial]


# ### Temporal ...
# extract spatial data
data = np.array(res['timestamps']).reshape(-1, 1)
metric = clust.metric_temporal()
# build a first set of spatial clusterings
# specify bandwidth (in seconds) to be used in the spatial kernel :
bandwidths = np.array([1, 2, 4, 8, 12, 24, 48, 72]) * 3600
R_temporal = cooper.compute_clusterings_at_scales(
    feature=data, bandwidths=bandwidths, metric=metric)

# extract the boundaries of every clustering
clust_temporal = [r['clusters_indices'] for r in R_temporal]


# ## Now try to fuse every possible clustering with cooper's technic
# concatenate every possible clusterings
clusterings = clust_spatial + clust_temporal
# now apply Cooper's technic to build the clustering that maximizes
# the average mutual information between every pre computed clusterings
print('Make this work in a reasonable time for N ~ 300 photos :)')
e, b = cooper.score_for_clustering(clusterings, N)
