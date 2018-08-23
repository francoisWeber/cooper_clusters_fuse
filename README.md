# README AS SHORT AS USEFUL

## Prerequisite :
- Python 3
- `git clone https://gitlab.cozycloud.cc/paul/clustering.git`
- `pip install numpy scipy pandas scikit-learn`
- `yarn global add local-static-server` to install a simple static file server
- `cd react_gallery && yarn` to visualize photos clustering in the browser
- `yarn global add https://github.com/cozy/ACH` to import photos from a Cozy

## Import GPS-tagged photos

### From a Cozy:
`python scripts/download_cozy_files_with_gps.py <cozy_url>`

### From a local directory:
`python scripts/extract_photos_with_gps.py <dir_path>`


## Display photo clusters in the browser :
- `python dbscan_spatial.py <dir_path>`
- into another terminal: `local-server <dir_path> 1234`
- into another terminal: `cd react_gallery && yarn start`

You should now been able to see your picture organized by events on `http://localhost:8080`
