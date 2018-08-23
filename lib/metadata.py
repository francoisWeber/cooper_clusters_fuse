# -*- coding: utf-8 -*-

from datetime import datetime
from PIL import Image

# The hexadecimal codes are the Tag IDs in EXIF format
# See https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html


def extract_exif_info(image_path):
    # GPS hexa :
    gps = 0x8825
    # timestamp hexa
    time = 0x9003
    # dimensions
    width_px = 0x0100
    height_px = 0x0101
    # try to extract them from image
    coords = None
    timestamp = None
    width = None
    height = None
    with Image.open(image_path) as img:
        exif = img._getexif()
        # track GPS info
        if gps in exif:
            gps_info = exif[gps]
            if 0x0002 in gps_info and 0x0004 in gps_info:
                lat_ref = gps_info[0x0001]
                lat = gps_info[0x0002]
                long_ref = gps_info[0x0003]
                long = gps_info[0x0004]
                latitude = dms_to_decimal(lat, lat_ref)
                longitude = dms_to_decimal(long, long_ref)
                coords = {
                    'lat': latitude,
                    'long': longitude
                }
        # track timestamp info
        if time in exif:
            date_info = exif[time]
            d = datetime.strptime(date_info, '%Y:%m:%d %H:%M:%S')
            timestamp = d.timestamp()
        # bonus : track dimensions (to be passed to React)
        if width_px in exif and height_px in exif:
            width = exif[width_px]
            height = exif[height_px]
    return coords, timestamp, width, height


def extract_gps_info(image_path):
    with Image.open(image_path) as img:
        # 0x8825 is the Tag ID for 'GPSInfo'
        if 0x8825 in img._getexif():
            gps_info = img._getexif()[0x8825]
            if 0x0002 in gps_info and 0x0004 in gps_info:
                lat_ref = gps_info[0x0001]
                lat = gps_info[0x0002]
                long_ref = gps_info[0x0003]
                long = gps_info[0x0004]
                gps_dms = {
                    'lat': lat,
                    'lat_ref': lat_ref,
                    'long': long,
                    'long_ref': long_ref,
                }
                latitude = dms_to_decimal(lat, lat_ref)
                longitude = dms_to_decimal(long, long_ref)
                gps = {
                    'lat': latitude,
                    'long': longitude
                }
                return gps
    return None


def dms_to_decimal(gps, orient):
    if len(gps) != 3:
        return None
    dms = []
    for c in gps:
        dms.append(c[0] / c[1])
    return convert_to_decimal(dms[0], dms[1], dms[2], orient)


def convert_to_decimal(degrees, minutes, seconds, direction):
    d = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    if direction == 'W' or direction == 'S':
        d *= -1
    return d
