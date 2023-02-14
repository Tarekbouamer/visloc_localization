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

        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load():
        pass

    def device(self):
        return self.device

    def close(self):
        self.hfile.close()


class KeypointsLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)

    def load_as_numpy(self, name):
        dset = self.hfile[name]['keypoints']

        keypoint = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')    
        
        return keypoint, uncertainty    
        
    def load(self, name):
        
        # 
        keypoint, uncertainty = self.load_as_numpy(name)

        #
        keypoint = torch.from_numpy(keypoint).float()
        keypoint = keypoint.to(self.device)

        if uncertainty is not None:
            uncertainty = torch.from_numpy(uncertainty).float()
            uncertainty = uncertainty.to(self.device)

        return keypoint, uncertainty


class GlobalFeaturesLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)
    
    def load_as_numpy(self, name):
        x = self.hfile[name]["features"]
        return x.__array__()

        
    def load(self, name):
        preds = {}
        keys = list(self.hfile[name].keys())

        for k in keys:
            v = self.hfile[name][k].__array__()
            v = torch.from_numpy(v).float()
            preds[k] = v.to(self.device)

        return preds


class LocalFeaturesLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)

    def load(self, name):

        preds = {}
        keys = list(self.hfile[name].keys())

        for k in keys:
            v = self.hfile[name][k].__array__()
            v = torch.from_numpy(v).float()
            preds[k] = v.to(self.device)

        return preds


class MatchesLoader(Loader):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)

    def load_as_numpy(self,
                      name0: str,
                      name1: str
                      ):

        pair_key, reversed = find_pair(self.hfile, name0, name1)

        matches = self.hfile[pair_key]['matches'].__array__()
        scores = self.hfile[pair_key]['scores'].__array__()

        idx = np.where(matches != -1)[0]
        matches = np.stack([idx, matches[idx]], -1)

        if reversed:
            matches = np.flip(matches, -1)

        scores = scores[idx]

        return matches, scores

    def load(self,
             name0: str,
             name1: str
             ):

        #
        matches, scores = self.load_as_numpy(name0, name1)

        matches = torch.from_numpy(matches).float()
        scores = torch.from_numpy(scores).float()

        matches = matches.to(self.device)
        scores = scores.to(self.device)

        return matches, scores
