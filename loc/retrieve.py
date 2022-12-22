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

from loc.utils.io import  find_unique_new_pairs

# logger
import logging
logger = logging.getLogger("loc")


def get_descriptors(desc_path, names, key='global_descriptor'):
    
    descs = []

    for n in names:
        with h5py.File(str(desc_path), 'r') as fd:
            x = fd[n][key].__array__()
            descs.append(fd[n][key].__array__())
            
    out = torch.from_numpy(np.stack(descs, 0)).float()
    
    return out


class Retrieval(object):
    
    def __init__(self, workspace, save_path, cfg):
        
        # cfg
        self.cfg    = cfg
        model_name  = self.cfg.retrieval.model_name
        num_topK    = self.cfg.retrieval.num_topK
       
        # extractor
        logger.info(f"init retrieval {model_name}")
        self.extractor = FeatureExtractor(model_name=model_name)
        
        #
        self.workspace  = workspace
        self.save_path  = save_path
        
        # paris path
        self.pairs_path = save_path /   Path('pairs'            + '_' + \
                                        str(model_name)         + '_' + \
                                        str(num_topK)           + '.txt') 

    
    def extract_images(self, images_path, split=None):
        
        images          = ImagesFromList(root=images_path, split=split, cfg=self.cfg)
        features_path   = Path(str(self.save_path) + '/' + str(split) + '_global_features' + '.h5')
        
        preds = self.extractor.extract_global(images, save_path=features_path, normalize=True, )
        
        return preds
    
    def extract_images_databse(self):
        
        logger.info(f"extract global features for databse images ")

        db_preds = self.extract_images(self.workspace, split="db")
        
        return db_preds

    def extract_images_queries(self):
        
        logger.info(f"extract global features for query images ")

        db_preds = self.extract_images(self.workspace, split="query")
        
        return db_preds    
    
    def __match(self, q_preds, db_preds):
        #
        q_descs  = q_preds["features"]
        db_descs = db_preds["features"]
        
        q_names  = q_preds["names"]
        db_names = db_preds["names"]
        
        # similarity
        scores = torch.mm(q_descs, db_descs.t())
        
        # search for num_topK images
        num_topK = self.cfg.retrieval.num_topK
        
        logger.info(f"retrive top {num_topK} images")   
        
        invalid = np.array(q_names)[:, None] == np.array(db_names)[None]   
        invalid = torch.from_numpy(invalid).to(scores.device)   

        invalid |= scores < 0
        scores.masked_fill_(invalid, float('-inf'))

        topk    = torch.topk(scores, k=num_topK, dim=1)
        
        indices = topk.indices.cpu().numpy()
        valid   = topk.values.isfinite().cpu().numpy() 
        
        # collect pairs 
        pairs = []
        for i, j in zip(*np.where(valid)):
            pairs.append((i, indices[i, j]))
    
        name_pairs = [(q_names[i], db_names[j]) for i, j in pairs]    
        
        assert len(name_pairs) > 0, "No matching pairs has been found! "
        
        return name_pairs
    
    def __remove_duplicates(self, pairs):
        return find_unique_new_pairs(pairs)
    
    def retrieve(self, remove_duplicates=True):
    
        # extract
        db_preds    = self.extract_images_databse()
        q_preds     = self.extract_images_queries()
        
        # match
        image_pairs = self.__match(q_preds, db_preds)
        
        # remove duplicates
        if remove_duplicates:
            image_pairs = self.__remove_duplicates(image_pairs)

        # save pairs
        with open(self.pairs_path, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in image_pairs))
        
        logger.info(f"{len(image_pairs)} pairs have been found, saved {self.pairs_path}")         
  
        return self.pairs_path
    
    
    
    
def do_retrieve(workspace, save_path, cfg):
    
    # save
    ret = Retrieval(workspace=workspace, 
                    save_path=save_path,
                    cfg=cfg)
    
    # retrieve
    image_pairs = ret.retrieve()   
       
    return image_pairs