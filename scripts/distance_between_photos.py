import sys
import os
base_dir = os.path.dirname(__file__) + '/../'
try:
    sys.path.insert(1, base_dir)
except BaseException:
    sys.path.insert(1, os.path.join(base_dir))
from lib.metadata import extract_gps_info
from lib.utils import geodesic_distance

# Compute the geodesic distance between two photos, based on the EXIF GPS info

if len(sys.argv) != 3:
    print("Usage: {} <photo1> <photo2>".format(sys.argv[0]))
    exit()

photo1_path = sys.argv[1]
photo2_path = sys.argv[2]

gps_photo1 = extract_gps_info(photo1_path)
gps_photo2 = extract_gps_info(photo2_path)
gps_photo1 = [gps for gps in gps_photo1.values()]
gps_photo2 = [gps for gps in gps_photo2.values()]

dist = geodesic_distance(gps_photo1, gps_photo2)

print("Distance between {} and {}: {:.3f} km".format(
    os.path.basename(photo1_path),
    os.path.basename(photo2_path),
    dist
    )
)
