# -*- coding: utf-8 -*-
import timeit
import os
from subprocess import run
from math import radians, cos, sin, asin, sqrt

# To use start_measure and stop_measure:
#   start = start_measure()
#   do stuff...
#   stop_measure(start, "my measure")


def start_measure():
    return timeit.default_timer()


def end_measure(start, label=""):
    stop = timeit.default_timer()
    if label != "":
        print("Time elapsed for {0:s}: {1:.2f}s".format(label, stop - start))
    return stop - start


def start_server(path):
    with open(os.devnull, 'w') as devnull:
        run(['local-server', path, '1234'], stdout=devnull)
        print("Local server running on port 1234...")


def show_distance_matrix(data):
    data = data.get_values()
    dists = np.zeros(shape=(data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dists[i, j] = geodesic_distance(data[i, :], data[j, :])
    plt.imshow(np.square(dists))
    plt.colorbar()
    plt.show()


def geodesic_distance(x1, x2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    INPUTS:
    - x1 : coordinate point having [lattitude, longitude]
    - x2 : same

    OUTPUT :
    - km : distance in kilometers with respect to the geodesic
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [x1[1], x1[0], x2[1], x2[0]])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371 * c
    return km
