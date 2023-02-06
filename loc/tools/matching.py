# logger
import logging

from functools import partial
from queue import Queue
from threading import Thread

from torch.utils.data import DataLoader
from loc.datasets import PairsDataset

from tqdm import tqdm

from loc.matchers import Matcher

from loc.utils.io import (remove_duplicate_pairs, 
                          read_pairs_list,
                          pairs2key)

from loc.utils.writers import MatchesWriter

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

        
def do_matching(pairs_path, src_path, dst_path, cfg=None, save_path=None, num_threads=4):
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
    pairs = read_pairs_list(pairs_path)
    pairs = remove_duplicate_pairs(pairs)

    if len(pairs) == 0:
        logger.error('no matches pairs found')
        return

    # pair dataset loader
    pair_dataset = PairsDataset(pairs=pairs, src_path=src_path, dst_path=dst_path)
    pair_loader  = DataLoader(pair_dataset, num_workers=16, batch_size=1, shuffle=False, pin_memory=True)
    
    logger.info("matching %s pairs", len(pair_dataset))  
    
    # matches writer
    writer = MatchesWriter(save_path=save_path)
    
    # workers
    writer_queue  = WorkQueue(partial(writer.write_matches), num_threads)
            
    # matcher
    matcher = Matcher(cfg=cfg)
    
    # run
    for _, (src_name, dst_name, data) in enumerate(tqdm(pair_loader, total=len(pair_loader))):

        # match
        preds = matcher.match_pair(data)
        
        # get key
        pair_key = pairs2key(src_name[0], dst_name[0])
        
        # put
        writer_queue.put((pair_key, preds))
    
    # collect workers    
    writer_queue.join()
    writer.close()

    #      
    logger.info("matches saved to %s", str(save_path) )
    
    return save_path
    
