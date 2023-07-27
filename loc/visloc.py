import argparse
# logger
from loguru import logger
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pycolmap
from tqdm import tqdm

from loc.solvers.pnp import (AbsolutePoseEstimationPoseLib,
                             AbsolutePoseEstimationPyColmap)
from loc.utils.io import (dump_logs,
                          read_pairs_dict, write_poses_txt)

from loguru import logger
class VisLoc(object):
    """_summary_

    Args:
        object (_type_): _description_
    """  
    def __init__(self, mapper, extractor, matcher, retrieval, cfg={}):
        
        self.cfg = cfg
        
        # 
        self.mapper     = mapper
        self.extractor  = extractor        
        self.matcher    = matcher
        self.retrieval  = retrieval
        
    def pairs_from_sfm(self):
        return self.mapper.covisible_pairs()
        
    def pairs_from_retrieval(self,):
        return self.retrieval.retrieve()
    
    def prepare_database(self):
        
        self.mapper.run_sfm()

        # 1. mapper build a map sfm :
        #       do mapping from scratch if not existing 
        #       import colmap model from workspace folder
         
        # 2. extract features:
        #       use visloc to extract local features     
        #    
        
        # 3. perform matching on database
        
        # 4. create database and create visloc model
                
        pass

    def geo_localization():
        
        # 1. perform features extraction
        
        # 2. match and search
        
        # 3. re-rank algo
        
        pass
    
    
    def localize():
        
        # 1. get 2D-3D correspondences
        
        # 2.  pose solver

        pass

          
    def run(self):
        
        self.prepare_database()
        
        # A. prepare database
        
        # B. perform image retrieval
        
        # C. pose estimation
          
        pass
        
        
