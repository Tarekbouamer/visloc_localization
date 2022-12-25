import collections.abc as collections
import inspect
# logger
import logging
import pprint
import sys
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import loc.matchers as matchers
from loc.utils.io import (find_unique_new_pairs, 
                          get_pairs_from_txt,
                          names_to_pair, 
                          names_to_pair_old, 
                          read_key_from_h5py)

logger = logging.getLogger("loc")
    
    
class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue      = Queue(num_threads)
        self.threads    = [Thread(target=self.thread_fn, args=(work_fn,)) 
                           for _ in range(num_threads)]
        
        for thread in self.threads:
            thread.start()

    def join(self):
        
        for thread in self.threads:
            self.queue.put(None)
        
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class PairsDataset(Dataset):
    """pair dataset

    Args:
        pairs (list): pairs list
        src_path (str): path to src image
        dst_path (str): path to dst image
    """    
    def __init__(self, pairs, src_path, dst_path):
        
        self.pairs = pairs
        
        self.src_path = src_path
        self.dst_path = dst_path

    def __getitem__(self, idx):
        
        data = {'src': {}, 'dst': {}}

        src_name, dst_name = self.pairs[idx]

        data['src_name'] = src_name
        data['dst_name'] = dst_name

        data['src'] = read_key_from_h5py(src_name, self.src_path)
        data['dst'] = read_key_from_h5py(dst_name, self.dst_path)
        
        return data

    def __len__(self):
        return len(self.pairs)


def matcher_writer(data, save_path):
    """matcher writer

    Args:
        data (tuple): input data (keys, m_dists, m_idxs)
        save_path (_type_): save_path to matches 
    """    
    
    #
    pair_key, m_dists, m_idxs = data

    #
    with h5py.File(str(save_path), 'a') as fd:
        
        #
        if pair_key in fd:
            del fd[pair_key]
                
        group = fd.create_group(pair_key)
        
        #
        m_idxs  = m_idxs.cpu().short().numpy()
        m_dists = m_dists.cpu().half().numpy()
                                
        group.create_dataset('matches', data=m_idxs )
        group.create_dataset('scores',  data=m_dists)               
        
        
def do_matching(pairs_path, src_path, dst_path, save_path=None, num_threads=4):
    """general matching 

    Args:
        pairs_path (str): pairs path
        src_path (str): src image features path
        dst_path (str): dst image features path
        save_path (str, optional): path to save matches. Defaults to None.
        num_threads (int, optional): number of workers. Defaults to 4.

    Returns:
        str: path to save matches 
    """    
    
    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # Load pairs 
    pairs = get_pairs_from_txt(pairs_path)
    pairs = find_unique_new_pairs(pairs)

    if len(pairs) == 0:
        logger.error('no matches pairs found')
        return

    # pair dataset loader
    pair_dataset = PairsDataset(pairs=pairs, src_path=src_path, dst_path=dst_path)
    pair_loader  = DataLoader(pair_dataset, num_workers=16, batch_size=1, shuffle=False, pin_memory=True)
    
    logger.info("matching %s pairs", len(pair_dataset))    
    
    # workers
    writer_queue  = WorkQueue(partial(matcher_writer, match_path=save_path), num_threads)
            
    # matcher
    matcher = matchers.MutualNearestNeighbor()
    
    # run
    for _, data in enumerate(tqdm(pair_loader, total=len(pair_loader))):
        
        src_name = data['src_name'][0]
        dst_name = data['dst_name'][0]
        
        src_desc = data['src']['descriptors'].cuda()
        dst_desc = data['dst']['descriptors'].cuda()

        # match
        matches_dists, matches_idxs = matcher(src_desc, dst_desc)
        
        # get key
        pair_key = names_to_pair(src_name, dst_name)
        writer_queue.put((pair_key, matches_dists, matches_idxs))
    
    # collect workers    
    writer_queue.join()

    #      
    logger.info("matches saved to %s", str(save_path) )
    
    return save_path
    
