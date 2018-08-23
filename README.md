# README AS SHORT AS USEFUL

## Goal :
We consider a set of personal photos (here ~ 300 items) with timestamp and GPS coordinates. We want to clusterize them into clusters representing "events of life". 
 
## Prerequisite :
- Python 3
- `pip install numpy scipy pandas scikit-learn`
- check you have `res.json` that contains informations about the photos 

## Clusterizing GPS-tagged photos
I imported ~330 photos from my cellphone. The goal of this script is to build a clustering of these photos using a method detailed in Paper 1 and Paper 2. 
More specifically, Paper 1 explains how to build a hierarchy of clusters with respect to one particular feature (here GPS coords and/or timestamp). Here we consider 2 kind of features : GPS coordinate and timestamps ; thus we will get 2 hierarchy of clusterings, each one based on one of these features.
Paper 2 then explains how to _fuse_ a set of clusterings into 1 unique clustering that is optimal in a certain sens : maximizing the average normalized mutual information (NMI) between every given clusterings.

## What is already implemented 
I already implemented the uselfull parts of Paper 1. The function `lib/cooper.py:compute_clusterings_at_scales()` computes a set of hierarchical clustering for a given features. 
So for 1 given features (either GPS or timestamp) it is already possible to build a set of clusters.

## Where implementation gets tricky
Paper 2 shows how to fuse a set of clusters. For that, we need 2 things :
- a criterion of how good would be an arbitrary extension of the current clustering. This is equation (12) which is implemented at `lib/cooper.py:nmi_criterion_between_indices()`.
- a way of incrementally building the optimal unique cluster wrt the criterion. This is equation (7) in which we plugged the criterion `C_{NMI}` instead of `C_{F}`.

The problem is : implenting equation (7) is _very hard_ ; the naive recursive implementation lasts hours. 
Other problem : the goal of that function is twofold :
- to evaluate the "cost" of a clustering for any number of clusters `K`
- to give the clustering that maximizes the NMI criterion for each `K`

## Help me ? :D
The `run_me.py` should perform every preliminary computations. It ends with the `cooper.score_for_clustering()` that is very slow and tricky to implement to give at one time :
- the optimal clusterings for each `K = 2, 3, ...`
- the costs related to these clusters