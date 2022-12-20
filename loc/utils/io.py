from collections import defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Union, Tuple


import contextlib
import io
import sys


import torch
import h5py
import pickle

import pycolmap
from loc import logger

# 
# logger
import logging
logger = logging.getLogger("loc")


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


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs

# h5py

def read_key_from_h5py(name, _path):
    data = {}
    with h5py.File(str(_path), 'r') as f:
            
        if name in f:
            g = f[name]
        else:
            logger.error(f'{name} not found in {_path}')
            
        for k, v in g.items():
            data[k] = torch.from_numpy(v.__array__()).float()

    return data

def get_keypoints(path, name, return_uncertainty=False):
    
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
    if return_uncertainty:
        return p, uncertainty
    return p


def get_matches(path, name0, name1):
    
    #
    with h5py.File(str(path), 'r', libver='latest') as hfile:
        
        pair, reverse = find_pair(hfile, name0, name1)
        
        matches = hfile[pair]['matches'].__array__()
        scores  = hfile[pair]['scores'].__array__()
    #
    idx     = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    
    #
    if reverse:
        matches = np.flip(matches, -1)
    
    #
    scores = scores[idx]
    
    return matches, scores


# parser
def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))

def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')

def find_pair(hfile: h5py.File, name0: str, name1: str):
    
    pair = names_to_pair(name0, name1)    

    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    
    if pair in hfile:
        return pair, True
    
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')
  
def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)

# pose
def dump_logs(logs, save_path):
    
    save_path = f'{save_path}_logs.pkl'

    logger.info(f'writing logs to {save_path} ')
    
    with open(save_path, 'wb') as f:
        pickle.dump(logs, f)
          
def write_poses_txt(poses, save_path):

    logger.info(f'writing poses to {save_path}...')

    with open(save_path, 'w') as f:
        for q in poses:
            
            name        = q.split('/')[-1]
            qvec, tvec  = poses[q]
            qvec        = ' '.join(map(str, qvec))
            tvec        = ' '.join(map(str, tvec))

            f.write(f'{name} {qvec} {tvec}\n')
            
# capture

class OutputCapture:
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                print('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()