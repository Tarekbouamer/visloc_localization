# logger
from pathlib import Path

from omegaconf import OmegaConf

import logging
logger = logging.getLogger("loc")


def make_aachen_cfg(cfg):
    
    meta = {
        'query': {
            'images':   "images/query/",
            'cameras':  'queries/*_time_queries_with_intrinsics.txt'
            },
        'db': { 
            'images':   "images/db/",
            'cameras':  None
            },
        
        'convert':{
            'type':         "nvm",
            'nvm_path':     cfg.workspace + "/"  +'3D-models/aachen_cvpr2018_db.nvm',
            'intrinsics':   cfg.workspace + "/"  + '3D-models/database_intrinsics.txt',
            'database':     cfg.workspace + "/"  + 'aachen.db',
        }
        
    } 
    
    return meta 
          
          
def make_config(name="default", cli_cfg={}) :
    
    # read cli cfg 
    # base cfg
    # dataset cfg --> TODO: to yml files for each dataset
    # merger by order 
    
    # load config file
    cfg = OmegaConf.load(cli_cfg.config)

    # make data confgi file
    if name == "default":
        data_cfg = {}
    elif name == "aachen":
        data_cfg = make_aachen_cfg(cli_cfg)
        
    # merge
    cfg = OmegaConf.merge(cfg, data_cfg, cli_cfg)
    
    return cfg
    