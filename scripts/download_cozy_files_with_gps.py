import json
import sys
import os
from subprocess import call

# This script downloads all the photos from a Cozy having GPS info
# It requires ACH for the export and download operations: https://github.com/cozy/ACH

OUTPUT_DIR='Photos_GPS'
JSON_FILE='files.json'

if len(sys.argv) < 2:
    print("Usage: {} <Cozy URL>".format(sys.argv[0]))
    exit()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

url = sys.argv[1]

# Export the JSON file containing all the Cozy's files info
call(["ACH", "export", "io.cozy.files", JSON_FILE, "-u", url])

with open(JSON_FILE) as f:
    data = json.load(f)

data = data['io.cozy.files']

gps_files = []
for d in data:
    if 'metadata' in d:
        exif = d['metadata']
        if exif.get('gps') is not None:
            gps_files.append({'id': d["_id"], 'name': d['name']})

print("{} files found with GPS metadata".format(len(gps_files)))

# Create the destination directory
base_url = url.split('https://')[1]
dest_dir = os.path.join(OUTPUT_DIR, base_url)
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
os.chdir(dest_dir)

# check which file has already been downloaded to avoid useless downloading
files_in_dest_dir = os.listdir('.')

# Download the files with GPS metadata
print("Let's save images into " + dest_dir)
already_here = 0
for f in gps_files:
    if f['name'] not in files_in_dest_dir:
        call(["ACH", "downloadFile", f['id'], "-u", url])
    else:
        already_here += 1
print(
        '{} files downloaded and {} ({}%) were already downloaded'.format
        (len(gps_files) - already_here,
            already_here,
            100*already_here/len(gps_files))
    )
