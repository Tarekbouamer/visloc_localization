from collections import defaultdict
from pathlib import Path
import numpy as np

import torch
import h5py

import pycolmap
from loc import logger

# 

def parse_name(name):
    return name.replace('/', '-')

def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def get_pairs_from_txt(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f.read().rstrip('\n').split('\n'):
            
            if len(line) == 0:
                continue
            
            q_name, db_name = line.split()
            pairs.append((q_name, db_name))
            
    return pairs   


def read_key_from_h5py(name, _path):
    
    data = {}
    
    with h5py.File(str(_path), 'r') as f:
            
        if name in f:
            g = f[name]
        else:
            logger.critical(f'{name} not found in {_path}')
            
        for k, v in g.items():
            data[k] = torch.from_numpy(v.__array__()).float()
    
    return data


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
  
  
def parse_retrieval_file(path):
    """
      Load retrieval pairs
    """
    ret = defaultdict(list)
    
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            
            if len(p) == 0:
                continue
            
            q, r = p.split()
            ret[q].append(r)
    
    return ret


  
  
