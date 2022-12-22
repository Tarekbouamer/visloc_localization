import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import os 

from loc.dataset        import ImagesFromList, ImagesTransform
from loc.extractors      import SuperPoint

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


class Extraction(object):

    
    def __init__(self, workspace, save_path, cfg):
        
        # cfg
        self.cfg = cfg
        
        # extractor
        logger.info(f"init feature extractor {cfg.extractor.name}")

        self.extractor = SuperPoint(config=self.cfg.extractor)
        
        #
        self.workspace  = workspace
        self.save_path  = save_path
    
    def extract_images(self, images_path, split=None):
        
        images          = ImagesFromList(root=images_path, split=split, cfg=self.cfg, gray=True)
        features_path   = Path(str(self.save_path) + '/' + str(split) + '_local_features' + '.h5')
        preds = self.extractor.extract_keypoints(images, save_path=features_path, normalize=False)
                
        return preds, features_path
    
    def extract_images_databse(self):
        
        logger.info(f"extract local features for databse images ")

        db_preds, db_path = self.extract_images(self.workspace, split="db")
        
        return db_preds, db_path

    def extract_images_queries(self):
        
        logger.info(f"extract local features for query images ")

        q_preds, q_path = self.extract_images(self.workspace, split="query")
        
        return q_preds, q_path    
    
    def extract(self):
    
        # extract
        db_preds, db_path = self.extract_images_databse()
        q_preds , q_path  = self.extract_images_queries()
        
        return db_path, q_path
    
    
    
    
def do_extraction(workspace, save_path, cfg):

    # save
    ext = Extraction(workspace=workspace, 
                     save_path=save_path,
                     cfg=cfg)
    
    # retrieve
    image_pairs = ext.extract()    
    
    return image_pairs