# -*- coding: utf-8 -*-

import os
import sys
import json

import pandas as pd
import numpy as np

from PIL import Image

from lib import metadata
from lib import config

# Returns the images from the given path


def list_images(path):
    images = []
    files = os.listdir(path)
    for f in files:
        if os.path.isfile(os.path.join(path, f)):
            split = os.path.splitext(f)
            if len(split) > 1:
                ext = split[1].lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    images.append(f)
    return images


def scan_images(dir_path, with_gps=False):
    print("Scan the images...")
    images = list_images(dir_path)

    timestamps = []
    filenames = []
    coords = []
    widths = []
    heights = []
    for img in images:
        img_path = os.path.join(dir_path, img)
        # Retrieve metadata
        meta = metadata.extract_exif_info(img_path)
        c = meta[0]
        t = meta[1]
        w = meta[2]
        h = meta[3]
        # filter them
        if t is not None:
            # Retrieve GPS info if needed
            if with_gps:
                if coords is not None:
                    coords.append(c)
            timestamps.append(t)
            filenames.append(img)
            widths.append(w)
            heights.append(h)

    # Sort data by timestamps
    timestamps = np.array(timestamps)
    timestamps_idx_order = np.argsort(timestamps)
    timestamps_ordered = [timestamps[t] for t in timestamps_idx_order]
    filenames_ordered = [filenames[t] for t in timestamps_idx_order]
    widths_ordered = [widths[t] for t in timestamps_idx_order]
    heights_ordered = [heights[t] for t in timestamps_idx_order]
    if with_gps:
        coords_ordered = [coords[t] for t in timestamps_idx_order]
    else:
        coords_ordered = None

    result_ordered = {
        'coords': coords_ordered,
        'timestamps': timestamps_ordered,
        'filenames': filenames_ordered,
        'widths': widths_ordered,
        'heights': heights_ordered
    }
    return result_ordered


def save_dbscan_results(labels, filenames_ordered):
    # Vector of booleans: true if the index starts a new cluster
    ruptures = label2change(labels)
    # A same spatial cluster can be splitted into several events
    n_events = len([i for i, r in enumerate(ruptures) if r]) + 1
    print("{} images grouped into {} clusters and {} events".format(
        len(labels),
        len(set(labels)),
        n_events))

    print("Write the results...")
    save_result_json(labels, ruptures, filenames_ordered)
    print("Done!")


def label2change(labels_vector):
    if isinstance(labels_vector, type([])):
        n = len(labels_vector)
    elif isinstance(labels_vector, type(np.empty(1))):
        n = labels_vector.shape[0]
    else:
        raise TypeError(
            'Cannot handle input type {}'.format(
                type(labels_vector)))
    prev_label = labels_vector[0]
    change_detected = np.zeros(n, dtype=np.bool)
    for (i, l) in enumerate(labels_vector):
        if l != prev_label:
            change_detected[i] = True
            prev_label = l
    return change_detected


def save_result_json(labels, rupture_vector, filenames):
    events_bundle = []
    photos_bundle = []
    prev_label = labels[0]
    for (label, rupture, photo) in zip(labels, rupture_vector, filenames):
        json_photo = {
            'src': 'http://127.0.0.1:1234/' + photo,
            'width': 4,
            'height': 3
        }
        if not rupture:
            photos_bundle.append(json_photo)
        else:
            event = {'photos': photos_bundle,
                     'cluster': str(prev_label)}
            events_bundle.append(event)
            photos_bundle = [json_photo]
        prev_label = label
    if len(photos_bundle) > 0:
        event = {'photos': photos_bundle,
                 'cluster': str(prev_label)}
        events_bundle.append(event)

    with open(config.OUTPUT_RAW_PATH, 'w') as f:
        json.dump(events_bundle, f, indent=4)
