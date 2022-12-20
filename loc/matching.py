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

from queue import Queue
from threading import Thread
from functools import partial

import loc.matchers as matchers
from torch.utils.data import Dataset, DataLoader

from loc.utils.io import names_to_pair, names_to_pair_old, get_pairs_from_txt, read_key_from_h5py, find_unique_new_pairs

# logger
import logging
logger = logging.getLogger("loc")
    
    
class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
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
    """"""
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


def writer_fn(data, match_path):
    
    #
    pair_key, matches_dists, matches_idxs = data

    #
    with h5py.File(str(match_path), 'a') as fd:
        #
        if pair_key in fd:
            del fd[pair_key]
                
        group = fd.create_group(pair_key)
        
        #
        matches_idxs    = matches_idxs.cpu().short().numpy()
        matches_dists   = matches_dists.cpu().half().numpy()
                                
        group.create_dataset('matches', data=matches_idxs   )
        group.create_dataset('scores',  data=matches_dists  )               
        
        
def do_matching(src_path, dst_path, pairs_path, output):
    
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
    writer_queue  = WorkQueue(partial(writer_fn, match_path=output), 16)
            
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
    logger.info("matches saved to %s", str(output) )
    
