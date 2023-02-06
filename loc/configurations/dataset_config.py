# logger
from pathlib import Path

from omegaconf import OmegaConf

import logging
logger = logging.getLogger("loc")


def make_aachen_cfg(args, cfg):
    
    meta = {
        'query': {
            'images':   "images/images_upright/query/",
            'cameras':  'queries/*_time_queries_with_intrinsics.txt'
            },
        'db': { 
            'images':   "images/images_upright/db/",
            'cameras':  None
            },
        
        'convert':{
            'type':         "nvm",
            'nvm_path':     args.workspace + "/"  +'3D-models/aachen_cvpr2018_db.nvm',
            'intrinsics':   args.workspace + "/"  + '3D-models/database_intrinsics.txt',
            'database':     args.workspace + "/"  + 'aachen.db',
        }
        
    } 
    
    return meta 
          
          
def make_config(name="default", args={}) :
    
    # load config file
    cfg = OmegaConf.load(args.config)
    
    # make data confgi file
    if name == "default":
        data_cfg = {}
    elif name == "aachen":
        data_cfg = make_aachen_cfg(args, cfg)
        
    # merge
    cfg = OmegaConf.merge(cfg, data_cfg)
    logger.info(OmegaConf.to_yaml(cfg))
    
    return cfg
    