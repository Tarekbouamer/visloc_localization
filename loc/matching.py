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


import loc.matchers as matchers

# logger
import logging
logger = logging.getLogger("loc")

   
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
       
        
def do_matching(src_path, dst_path, pairs_path, output):

    # device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # Load pairs 
    pairs = get_pairs_from_txt(pairs_path)

    if len(pairs) == 0:
        logger.error('No Matches pairs found.')
        return

    logger.info("Match %s pairs", len(pairs))    
              
    # matcher
    matcher = matchers.MutualNearestNeighbor()
    
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
    
