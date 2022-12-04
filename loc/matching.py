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

from loc.utils.io import names_to_pair, get_pairs_from_txt, read_key_from_h5py

# logger
import logging
logger = logging.getLogger("loc")
    
        
def do_matching(src_path, dst_path, pairs_path, output):
    
    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # Load pairs 
    pairs = get_pairs_from_txt(pairs_path)

    if len(pairs) == 0:
        logger.error('No Matches pairs found.')
        return

    logger.info("matching %s pairs", len(pairs))    
              
    # matcher
    matcher = matchers.MutualNearestNeighbor()
    
    # run
    for it, (src_name, dst_name) in enumerate(tqdm(pairs, total=len(pairs))):
        
        #
        data = {'src': {}, 'dst': {}}
        
        # src 
        data['src'] = read_key_from_h5py(src_name, src_path)
        data['dst'] = read_key_from_h5py(dst_name, dst_path)

        src_desc = data['src']['descriptors']
        dst_desc = data['dst']['descriptors']
        
        if src_desc.shape[-1] != dst_desc.shape[-1]:
            src_desc = src_desc.T
            dst_desc = dst_desc.T
            
        # match
        matches_dists, matches_idxs = matcher(src_desc, dst_desc)
        
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
    logger.info("matches saved to %s", str(output) )
    
