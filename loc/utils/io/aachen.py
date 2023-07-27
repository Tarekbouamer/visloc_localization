from collections import defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Union, Tuple


import pycolmap

# 
# logger
from loguru import logger
from loguru import logger
def load_aachen_intrinsics(paths):
    """
      Load Aachen cameras from txt files
    """
    
    files = list(Path(paths.parent).glob(paths.name))
    
    assert len(files) > 0
    
    cameras = {}
    for file in files:
        parse_name_to_cameras_file(file, cameras)
  
    return cameras
  
  
def parse_name_to_cameras_file(path, cameras):
      
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            
            if len(line) == 0 or line[0] == '#':
                continue
            
            # unpack
            name, model, width, height, *params = line.split()  
            params = np.array(params, float)
            
            # Colmap
            cam = pycolmap.Camera(model, int(width), int(height), params)
            cameras[name] = cam
          
    return cameras