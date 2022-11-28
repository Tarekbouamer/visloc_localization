import torch
import sys
from abc import ABCMeta, abstractmethod
from torch import nn
from copy import copy
import inspect
from typing import Optional, Tuple
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
import collections.abc as collections
from tqdm import tqdm
import h5py
import torch

import numpy as np

import os

from collections import OrderedDict, defaultdict

# logger
import logging
logger = logging.getLogger("loc")


class BaseMatcher(nn.Module):
    def __init__(self, macther_type='base', **kwargs):
        super().__init__()
      
        self.macther_type = macther_type 
        
        logger.info(f"Matcher type ( {self.macther_type} ) is initialized")

    def forward(self, desc1, desc2):
        """
            desc1:        torch.Tensor    Batch of descriptors (B1, D)
            desc2:        torch.Tensor    Batch of descriptors (B2, D)
            
            distance:     torch.Tensor    Descriptor distance of matching descriptors
            indexes:      Long tensor     indexes of matching descriptors in desc1 and desc2,
        """
        raise NotImplementedError


class NearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='nearest_neighbor'):
        super().__init__(matcher_type)
      
    def forward(self, desc1, desc2):
        """"""
        cost = torch.cdist(desc1, desc2)
        
        # match distances 
        match_dists, idxs_in_2 = torch.min(cost, dim=1)
        idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
        
        # match indices 
        matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    
        return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
      
      
class MutualNearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='mutual_nearest_neighbor'):
        super().__init__(matcher_type)
      
    def forward(self, desc1, desc2):
        """"""
        #
        cost = torch.cdist(desc1, desc2)
        
        ms = min(cost.size(0), cost.size(1))
        
        #
        match_dists,  idxs_in_2 = torch.min(cost, dim=1)
        match_dists2, idxs_in_1 = torch.min(cost, dim=0)
        
        minsize_idxs = torch.arange(ms, device=cost.device)

        #
        if cost.size(0) <= cost.size(1):
            mutual_nns    = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
            matches_idxs  = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
            match_dists   = match_dists[mutual_nns]
        else:
            mutual_nns    = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
            matches_idxs  = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
            match_dists   = match_dists2[mutual_nns]
        
        return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
      

class NearestNeighborRatio(BaseMatcher):
    def __init__(self, matcher_type='nearest_neighbor_ratio', thd=0.):
        super().__init__(matcher_type)
        
        self.thd = thd
      
    def forward(self, desc1, desc2):
        """"""
        cost = torch.cdist(desc1, desc2)
        
        #
        vals, idxs_in_2 = torch.topk(cost, 2, dim=1, largest=False)
        ratio   = vals[:, 0] / vals[:, 1]
        
        # mask
        mask    = ratio <= self.thd
        
        # distance 
        match_dists = ratio[mask]
        
        idxs_in1  = torch.arange(0, idxs_in_2.size(0), device=cost.device)[mask]
        idxs_in_2 = idxs_in_2[:, 0][mask]
        
        # matches 
        matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
        
        return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
      