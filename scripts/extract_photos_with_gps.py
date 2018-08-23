import sys
import os
import PIL.Image
from shutil import copy

# This script takes into input a folder of photos and copy those having
# GPS metadata into the output folder

OUTPUT_DIR='Photos_GPS'

if len(sys.argv) < 2:
    print("Usage: {} <photo_dir_path>".format(sys.argv[0]))
    exit()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

dir_path = sys.argv[1]
images = os.listdir(dir_path)

gps_files = []
for img_name in images:
    img_path = os.path.join(dir_path, img_name)
    try:
        with PIL.Image.open(img_path) as img:
            if img._getexif() == None:
                continue
            # 0x8825 is the Tag ID for 'GPSInfo' in EXIF format
            # See https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
            if 0x8825 in img._getexif():
                gps_info = img._getexif()[0x8825]
                # 0x0002 is the Tag ID for 'GPSLatitude'
                # 0x0004 is the Tag ID for 'GPSLongitude'
                if 0x0002 in gps_info and 0x0004 in gps_info:
                    print("{} has GPS Info!".format(img_name))
                    gps_files.append(img_path)

    except IOError:
        # This is probably not an image: skip
        continue

print("{} images scanned".format(len(images)))

if len(gps_files) < 1:
    print("No image with GPS metadata found")
    exit()

dest_dir = os.path.join(OUTPUT_DIR, os.path.basename(os.path.normpath(dir_path)))
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for file in gps_files:
    copy(file, dest_dir)

print("{} images found with GPS metadata and copied to {}".format(len(gps_files), dest_dir))
