import h5py

from pathlib import Path

# path to hpy file
file_path = Path("/media/loc/D0AE6539AE65196C/VisualLocalization2020/aachen/visloc/sfm_matches.h5")

# open file
with h5py.File(file_path, 'r') as f:
    
    # print keys
    print(list(f["db-749.jpg"]["db-753.jpg"].keys()))
    