# logger
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch

from loc.utils.io import find_pair

logger = logging.getLogger("loc")


class Loader:
    def __init__(self,
                 save_path: Path
                 ) -> None:
        # save file
        self.save_path = save_path

        # writer
        self.hfile = h5py.File(str(save_path), 'r', libver='latest')
    
    def load():
        pass
    
    def close(self):
        self.hfile.close()


class KeypointsLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)
        
    def load_keypoints(self, name):
      
        dset = self.hfile[name]['keypoints']
        
        keypoint    = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
        
        return keypoint, uncertainty


class MatchesLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)

    def load_matches(self, 
                     name0: str, 
                     name1: str
                     ):

        pair_key, reversed = find_pair(self.hfile, name0, name1)

        # TODO: read all keys in field and remove hard coded keys
        matches = self.hfile[pair_key]['matches'].__array__()
        scores  = self.hfile[pair_key]['scores'].__array__()

        idx = np.where(matches != -1)[0]
        matches = np.stack([idx, matches[idx]], -1)

        if reversed:
            matches = np.flip(matches, -1)

        scores = scores[idx]

        return matches, scores
