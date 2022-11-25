import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import os 


def get_descriptors(desc_path, names, key='global_descriptor'):
    
    descs = []

    for n in names:
        with h5py.File(str(desc_path), 'r') as fd:
            descs.append(fd[n][key].__array__())
            
    out = torch.from_numpy(np.stack(descs, 0)).float()
    
    return out


def do_retrieve(meta, output, topK=5, override=False, logger=None):

    if logger:
        logger.info("Retrive top %s images", topK)   
            
    # 
    if output.exists():
        if override:
            os.remove(output)
        else:
            return output

    # device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get descs
    db_descs    = get_descriptors(desc_path=meta["db"]["path"], 
                                  names=meta["db"]["names"]).to(device=device)
    
    q_descs     = get_descriptors(desc_path=meta["query"]["path"], 
                                  names=meta["query"]["names"]).to(device=device)
    
  
    # Compute dot product scores and ranks
    scores = torch.mm(q_descs, db_descs.t())

    q_names   = meta["query"]["names"]
    db_names  = meta["db"]["names"]
        
    invalid = np.array(q_names)[:, None] == np.array(db_names)[None]   
    invalid = torch.from_numpy(invalid).to(scores.device)    
    invalid |= scores < 0
    scores.masked_fill_(invalid, float('-inf'))

    topk    = torch.topk(scores, k=topK, dim=1)
    
    indices = topk.indices.cpu().numpy()
    valid   = topk.values.isfinite().cpu().numpy() 
    
    # Find pairs 
    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    
    name_pairs = [(q_names[i], db_names[j]) for i, j in pairs]    
    
    assert len(name_pairs) > 0, "No matching pairs has been found! "
    
    if logger:
        logger.info("%s pairs have been found", len(name_pairs))         
    
    # Save
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in name_pairs))