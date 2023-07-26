from typing import Tuple
from pathlib import Path

from omegaconf import OmegaConf

import logging
from loguru import logger
DEFAULT_CFG = "loc/configurations/default.yml"

def make_aachen_cfg(args, cfg={}):
    
    data_cfg = {
        'query': {
            'images':   "images/query/",
            'cameras':  'queries/*_time_queries_with_intrinsics.txt'},
        'db': { 
            'images':   "images/db/",
            'cameras':  None}
        } 
    # 
    cfg = OmegaConf.merge(cfg, data_cfg)
    
    #
    args.type       = "nvm"
    args.model      = args.workspace / '3D-models/aachen_cvpr2018_db.nvm'
    args.intrinsics = args.workspace / '3D-models/database_intrinsics.txt'
    args.database   = args.workspace / 'aachen.db'
    args.save_path  = args.workspace / 'mapper'
    
    #
    return args, cfg
          
          
def make_workspace(args) -> None :
    """make workspace paths

    Args:
        args : arguments

    Returns:
        _type_: arguments
    """    

    # workspace
    args.workspace = Path(args.workspace)
    logger.info(f"workspace {args.workspace}")

    # images
    args.images_path = args.workspace / 'images'
    args.images_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"images {args.images_path}")
    
    # visloc
    args.visloc_path = args.workspace / 'visloc'
    args.visloc_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"visloc {args.visloc_path}")

    # mapper
    args.mapper_path = args.workspace / 'mapper'
    args.mapper_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"mapper {args.mapper_path}")

    return args

         
def make_config(args) -> Tuple:
    """get configuration and make necessary paths

    Args:
        args : arguments

    Returns:
        Tuple: arguments and configurations
    """    
    
    # default cfg
    default_cfg = OmegaConf.load(args.config)

    # make paths
    args = make_workspace(args)
    
    # dataset cfg
    if args.dataset == "aachen":
        args, cfg = make_aachen_cfg(args, default_cfg)
    
    else:
        cfg = default_cfg
        
    # 
    logger.info(f"config {OmegaConf.to_yaml(cfg)}")

    return args, cfg
    