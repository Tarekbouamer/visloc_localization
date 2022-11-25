import argparse
from asyncio.log import logger
from this import s
from tkinter import NONE
import torch
from typing import Dict, List, Union, Optional
import h5py
import cv2
import numpy as np
from tqdm import tqdm
import pprint
from enum import Enum

import os 
from pathlib import Path

# Core
from core.utils.configurations import make_config

# Visloc
from loc.configurations import DEFAULTS as DEFAULT_CONFIGS
from loc.config import *
 
# Config    
def import_from(module, name, method=None):
    try:      
        imported_module = __import__(module, fromlist=[name])
        imported_name = getattr(imported_module, name)
        
        if method is None: 
            return imported_name
        else:
            return getattr(imported_name, method) 
    except: 
        if method is not None: 
            name = name + '.' + method 
        print('WARNING: cannot import ' + name + ' from ' + module + ', check the file TROUBLESHOOTING.md')    
        return None 


# Nets
# GF_NetFeatures         = import_from(  'loc.models',    'GF_NetFeatures'       )         
# SuperPointFeatures     = import_from(  'loc.models',    'SuperPointFeatures'   )   

from loc.models import GF_NetFeatures
from loc.models import SuperPointFeatures

# Manager
class FeatureManager(object):
    
    def __init__(self, config, retrieval_type, detector_type, descriptor_type, logger=None, **varargs):
        
        self.detector_type      = detector_type
        self.descriptor_type    = descriptor_type
        self.retrieval_type     = retrieval_type
        
        self.config = config
        
        # Retrievals
        if self.retrieval_type == RetrievalTypes.GF_NET:
            self.image_retrieval    = GF_NetFeatures(logger=logger)
            
        # Detectors
        if self.detector_type == DetectorTypes.SUPERPOINT:  
            self.feature_detector   = SuperPointFeatures(logger=logger)  

        # Descriptors
        if self.descriptor_type == DescriptorTypes.SUPERPOINT:
            if self.detector_type != DetectorTypes.SUPERPOINT: 
                raise ValueError("select SUPERPOINT as both descriptor and detector!")
            self.feature_descriptor = self.feature_detector
        
        
        #    
        if logger:    
            logger.info("Feature Manager init Done")
    
    def save_features(self, item, preds, features_path, logger=None):
        
        item_name = item["img_name"]
                        
        # Scale keypoinst
        if 'keypoints' in preds:

            size    = np.array(item['img'].shape[:2][::-1])
            scales  = (item['original_size'] / size).astype(np.float32)

            preds['keypoints']  = (preds['keypoints'] + .5) * scales[None] - .5     
            uncertainty         = getattr(self, 'detection_noise', 1) * scales.mean()
            
        # Save features      
        with h5py.File(features_path, 'a') as fd:
            try:
                grp = fd.create_group(item_name)
                for k, v in preds.items():
                    grp.create_dataset(k, data=v)
                
                if 'keypoints' in preds:
                    grp['keypoints'].attrs['uncertainty'] = uncertainty

            except OSError as error:
                raise error
            
    # Compute the image global descriptor
    def extract_image_descriptor(self, dataset, path, override=True, logger=None):
        
        features_path = Path(str(path) + '_' + 'global_features.h5')
        
        if features_path.exists():
            if override:
                os.remove(features_path)
            else:
                return features_path
        
        if logger:
            logger.info("Extract global features" )
        
        for item in tqdm(dataset, total= len(dataset)):
            # run
            preds = self.image_retrieval(item["img"])
            preds = {k: v[0].cpu().numpy() for k, v in preds.items()}
            
            # save
            self.save_features(item, preds, features_path, logger)
            
            # del
            del preds

        if logger:
            logger.info("Save global features %s", features_path)
        
        return features_path
    
    def extract_local_features(self, dataset, path, override=True, logger=None):
        
        features_path = Path(str(path) + '_' + 'local_features.h5')

        if features_path.exists():
            if override:
                os.remove(features_path)
            else:
                return features_path
                    
        if logger:
            logger.info("Extract local features ")
        
        for item in tqdm(dataset, total= len(dataset)):
            
            # run
            preds = self.feature_detector(item["img"])
            preds = {k: v[0].cpu().numpy() for k, v in preds.items()}
            
            # save
            self.save_features(item, preds, features_path, logger)
            
            # del
            del preds

        if logger:
            logger.info("Save local features %s", features_path)
        
        return features_path    
    
    def extract(self, dataset, path, override=True, logger=None):
        
        features_path = Path(str(path) + '_' + 'global_and_local_features.h5')

        if features_path.exists():
            if override:
                os.remove(features_path)
            else:
                return features_path
             
        if logger:
            logger.info("Extract global & local features ")
        
        for item in tqdm(dataset, total= len(dataset)):
            
            # run
            preds_global = self.image_retrieval(item["img"])                           
            preds_local  = self.feature_detector(item["img"])
            
            preds = { **preds_global, **preds_local} 

            preds = {k: v[0].cpu().numpy() for k, v in preds.items()}
            
            # save
            self.save_features(item, preds, features_path, logger)
            
            # del
            del preds

        if logger:
            logger.info("Save global & local features %s", features_path)
        
        return features_path
    