import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import os 

from loc.dataset        import ImagesFromList, ImagesTransform
from loc.extractors     import FeatureExtractor

# logger
import logging
logger = logging.getLogger("loc")


def get_descriptors(desc_path, names, key='global_descriptor'):
    
    descs = []

    for n in names:
        with h5py.File(str(desc_path), 'r') as fd:
            x = fd[n][key].__array__()
            print(x.shape)
            descs.append(fd[n][key].__array__())
            
    out = torch.from_numpy(np.stack(descs, 0)).float()
    
    return out


def do_retrieve(dataset, data_config, outputs, topK=5):

    # model type
    retrieval_model_name = 'sfm_resnet50_gem_2048'
    
    # extractor
    logger.info(f"extract global features using {retrieval_model_name}")
    extractor = FeatureExtractor(model_name=retrieval_model_name)
                    
    # query images 
    image_set = ImagesFromList(root=dataset/data_config['query']["images"], split='query', max_size=400)
    save_path = Path(str(outputs) + '/' + str('query') + '_global' + '.h5')
    preds     = extractor.extract_global(image_set, save_path=save_path, normalize=True)
    q_descs   = preds["features"]
    q_names   = preds["names"]

    # db images 
    image_set = ImagesFromList(root=dataset/data_config['db']["images"], split='db', max_size=400)
    save_path = Path(str(outputs) + '/' + str('db') + '_global' + '.h5')
    preds     = extractor.extract_global(image_set, save_path=save_path, normalize=True)   
    db_descs  = preds["features"]
    db_names  = preds["names"]
    
    # compute dot product scores and ranks
    scores = torch.mm(q_descs, db_descs.t())
    
    # search for topK images
    logger.info("retrive top %s images", topK)   
    
    invalid = np.array(q_names)[:, None] == np.array(db_names)[None]   
    invalid = torch.from_numpy(invalid).to(scores.device)   

    invalid |= scores < 0
    scores.masked_fill_(invalid, float('-inf'))

    topk    = torch.topk(scores, k=topK, dim=1)
    
    indices = topk.indices.cpu().numpy()
    valid   = topk.values.isfinite().cpu().numpy() 
    
    # collect pairs 
    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    
    name_pairs = [(q_names[i], db_names[j]) for i, j in pairs]    
    
    assert len(name_pairs) > 0, "No matching pairs has been found! "
    
    # save
    loc_pairs_path = outputs / Path('pairs' + '_' +  str(retrieval_model_name) + '_' + str(topK)  + '.txt') 

    with open(loc_pairs_path, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in name_pairs))
    
    #  
    logger.info(f"{len(name_pairs)} pairs have been found, saved {loc_pairs_path}")         
  
    return loc_pairs_path