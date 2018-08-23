import PIL.Image
from PIL.ExifTags import GPSTAGS
import sys
import os

# This script takes into input a photo and returns the GPS info, if any

if len(sys.argv) < 2:
    print "Usage: {} <image_path>".format(sys.argv[0])
    exit()

img_path = sys.argv[1]

if not os.path.exists(img_path):
    print "No file found at {}".format(img_path)
    exit()

with PIL.Image.open(img_path) as img:
    if img._getexif() == None:
        print "No EXIF info"

    # 0x8825 is the Tag ID for 'GPSInfo' in EXIF format
    # See https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
    if 0x8825 in img._getexif():
        gps_info = img._getexif()[0x8825]
        # 0x0002 is the Tag ID for 'GPSLatitude'
        # 0x0004 is the Tag ID for 'GPSLongitude'
        if 0x0002 in gps_info and 0x0004 in gps_info:
            print "{} has GPS Info!".format(os.path.basename(img_path))
            print "{}: {}".format(GPSTAGS[0x0002], gps_info[0x0002])
            print "{}: {}".format(GPSTAGS[0x0004], gps_info[0x0004])
        else:
            print "No GPS Info"
    else:
        print "No GPS Info"
