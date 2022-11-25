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

class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        self.conf = conf = {**self.default_conf, **conf}
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        """Check the data and call the _forward method of the child model."""
        for key in self.required_inputs:
            assert key in data, 'Missing key {} in data'.format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        """To be implemented by the child class."""
        raise NotImplementedError

def find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new
  
  
class NearestNeighbor(BaseModel):
    default_conf = {
        'ratio_threshold': None,
        'distance_threshold': None,
        'do_mutual_check': True,
    }
    required_inputs = ['descriptors0', 'descriptors1']

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data['descriptors0'].size(-1) == 0 or data['descriptors1'].size(-1) == 0:
            matches0 = torch.full(
                data['descriptors0'].shape[:2], -1,
                device=data['descriptors0'].device)
            return {
                'matches0': matches0,
                'matching_scores0': torch.zeros_like(matches0)
            }
        ratio_threshold = self.conf['ratio_threshold']
        if data['descriptors0'].size(-1) == 1 or data['descriptors1'].size(-1) == 1:
            ratio_threshold = None
        sim = torch.einsum(
            'bdn,bdm->bnm', data['descriptors0'], data['descriptors1'])
        matches0, scores0 = find_nn(
            sim, ratio_threshold, self.conf['distance_threshold'])
        if self.conf['do_mutual_check']:
            matches1, scores1 = find_nn(
                sim.transpose(1, 2), ratio_threshold,
                self.conf['distance_threshold'])
            matches0 = mutual_check(matches0, matches1)
        return {
            'matches0': matches0,
            'matching_scores0': scores0,
        }


def match_nn(desc1, desc2, dm= None):
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.
    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.
    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.
    Returns:
        - Descriptor distance of matching descriptors, shape of :math:`(B1, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B1, 2)`.
    """
    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    idxs_in1: torch.Tensor = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
    matches_idxs: torch.Tensor = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_mnn(desc1, desc2, dm=None):
    """Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.
    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.
    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.
    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2, shape of :math:`(B3, 2)`,
          where 0 <= B3 <= min(B1, B2)
    """

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    ms = min(dm.size(0), dm.size(1))
    match_dists, idxs_in_2 = torch.min(dm, dim=1)
    match_dists2, idxs_in_1 = torch.min(dm, dim=0)
    minsize_idxs = torch.arange(ms, device=dm.device)

    if dm.size(0) <= dm.size(1):
        mutual_nns = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
        matches_idxs = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists[mutual_nns]
    else:
        mutual_nns = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        matches_idxs = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
        match_dists = match_dists2[mutual_nns]
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_snn(desc1, desc2, th=0.7, dm=None):
    """Function, which finds nearest neighbors in desc2 for each vector in desc1.
    The method satisfies first to second nearest neighbor distance <= th.
    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.
    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.
    Return:
        - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2. Shape: :math:`(B3, 2)`,
          where 0 <= B3 <= B1.
    """
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    vals, idxs_in_2 = torch.topk(dm, 2, dim=1, largest=False)
    ratio = vals[:, 0] / vals[:, 1]
    mask = ratio <= th
    match_dists = ratio[mask]
    idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=dm.device)[mask]
    idxs_in_2 = idxs_in_2[:, 0][mask]
    matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)


def match_smnn(desc1, desc2, th: float=0.7, dm=None):
    """Function, which finds mutual nearest neighbors in desc2 for each vector in desc1.
    the method satisfies first to second nearest neighbor distance <= th.
    If the distance matrix dm is not provided, :py:func:`torch.cdist` is used.
    Args:
        desc1: Batch of descriptors of a shape :math:`(B1, D)`.
        desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        th: distance ratio threshold.
        dm: Tensor containing the distances from each descriptor in desc1
          to each descriptor in desc2, shape of :math:`(B1, B2)`.
    Return:
        - Descriptor distance of matching descriptors, shape of. :math:`(B3, 1)`.
        - Long tensor indexes of matching descriptors in desc1 and desc2,
          shape of :math:`(B3, 2)` where 0 <= B3 <= B1.
    """

    if desc1.shape[0] < 2:
        raise AssertionError
    if desc2.shape[0] < 2:
        raise AssertionError

    if dm is None:
        dm = torch.cdist(desc1, desc2)
    else:
        if not ((dm.size(0) == desc1.size(0)) and (dm.size(1) == desc2.size(0))):
            raise AssertionError

    dists1, idx1 = match_snn(desc1, desc2, th, dm)
    dists2, idx2 = match_snn(desc2, desc1, th, dm.t())

    if len(dists2) > 0 and len(dists1) > 0:
        idx2 = idx2.flip(1)
        idxs_dm = torch.cdist(idx1.float(), idx2.float(), p=1.0)
        mutual_idxs1 = idxs_dm.min(dim=1)[0] < 1e-8
        mutual_idxs2 = idxs_dm.min(dim=0)[0] < 1e-8
        good_idxs1 = idx1[mutual_idxs1.view(-1)]
        good_idxs2 = idx2[mutual_idxs2.view(-1)]
        dists1_good = dists1[mutual_idxs1.view(-1)]
        dists2_good = dists2[mutual_idxs2.view(-1)]
        _, idx_upl1 = torch.sort(good_idxs1[:, 0])
        _, idx_upl2 = torch.sort(good_idxs2[:, 0])
        good_idxs1 = good_idxs1[idx_upl1]
        match_dists = torch.max(dists1_good[idx_upl1], dists2_good[idx_upl2])
        matches_idxs = good_idxs1
    else:
        matches_idxs, match_dists = torch.empty(0, 2, device=dm.device), torch.empty(0, 1, device=dm.device)
    return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
  
  
class DescriptorMatcher(nn.Module):
    known_modes = ['nn', 'mnn', 'snn', 'smnn']

    def __init__(self, match_mode, thd=0.7, logger=None):
        super().__init__()
        
        if match_mode not in self.known_modes:
            raise NotImplementedError(f"{match_mode} is not supported. Try one of {self.known_modes}")
        
        self.match_mode = match_mode
        self.thd = thd
        
        if logger:
            logger.info(f"Matcher type ( {self.match_mode} ) is initialized")

    def forward(self, desc1, desc2):
        """
        Args:
            desc1: Batch of descriptors of a shape :math:`(B1, D)`.
            desc2: Batch of descriptors of a shape :math:`(B2, D)`.
        Return:
            - Descriptor distance of matching descriptors, shape of :math:`(B3, 1)`.
            - Long tensor indexes of matching descriptors in desc1 and desc2,
                shape of :math:`(B3, 2)` where :math:`0 <= B3 <= B1`.
        """
        if self.match_mode == 'nn':
            out = match_nn(desc1, desc2)
        elif self.match_mode == 'mnn':
            out = match_mnn(desc1, desc2)
        elif self.match_mode == 'snn':
            out = match_snn(desc1, desc2, self.thd)
        elif self.match_mode == 'smnn':
            out = match_smnn(desc1, desc2, self.thd)
        else:
            raise NotImplementedError
        return out
     
    
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
       
        
def do_matching(src_path, dst_path, pairs_path, output, override=False, logger=None):

    # device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #
    if output.exists():
        if override:
            os.remove(output)
        else:
            return output
    
    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # Load pairs 
    pairs = get_pairs_from_txt(pairs_path)

    if len(pairs) == 0:
        logger.error('No Matches pairs found.')
        return

    if logger:
        logger.info("Match %s pairs", len(pairs))    
              
    # Init matcher
    matcher = DescriptorMatcher(match_mode='smnn', logger=logger)
    
    # run
    for (src_name, dst_name) in tqdm(pairs, total=len(pairs)):
        data = {'src': {}, 'dst': {}}
        
        # query 
        with h5py.File(str(src_path), 'r') as fq:
            group = fq[src_name]
            
            for key, v in group.items():
                data["src"][key] = torch.from_numpy(v.__array__()).float().to(device)
  
        # db    
        with h5py.File(str(dst_path), 'r') as fdb:
            group = fdb[dst_name]
            
            for key, v in group.items():
                data["dst"][key] = torch.from_numpy(v.__array__()).float().to(device)
 
        # Match
        with torch.no_grad():
             matches_dists, matches_idxs = matcher(  desc1=data['src']['descriptors'],   desc2=data['dst']['descriptors'])
        
        # Get key
        pair_key = names_to_pair(src_name, dst_name)
        
        # Save
        with h5py.File(str(output), 'a') as fd:
            
            if pair_key in fd:
                del fd[pair_key]
                
            group = fd.create_group(pair_key)
            
            matches_idxs    = matches_idxs.cpu().short().numpy()
            matches_dists   = matches_dists.cpu().half().numpy()
                        
            group.create_dataset('matches', data=matches_idxs   )
            group.create_dataset('scores',  data=matches_dists  )

    #      
    if logger:
        logger.info("Matches saved to %s", str(output) )
    
