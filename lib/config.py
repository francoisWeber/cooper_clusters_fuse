import json
import os

base_dir = os.path.dirname(__file__) + '/../'
config_path = os.path.join(base_dir, 'config.json')
params = json.load(open(config_path))

output_dir = params["output_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

OUTPUT_RAW_PATH = os.path.join(output_dir, params["output_raw"])
OUTPUT_IMG_PATH = os.path.join(output_dir, params["output_img"])
