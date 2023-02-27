import logging
from typing import Dict
logger = logging.getLogger("loc")
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from loc.utils.colmap.colmap_nvm import main as colmap_from_nvm


def convert_model_to_colmap(args, cfg):
    
    #
    logger.info(f"convert 3d model from {cfg.convert.type} to colmap format")

    # nvm -> colmap
    if cfg.convert.type == "nvm":
        colmap_from_nvm(nvm=Path(cfg.convert.nvm_path),
                        intrinsics=Path(cfg.convert.intrinsics),
                        database=Path(cfg.convert.database),
                        output=Path(cfg.mapper_path)) 
        
def run_model_conversion(args, cfg={}):
    
    logger.info(f"convert model from {args.type} to colmap format")

    if args.type == "nvm":
        colmap_from_nvm(nvm=args.model, 
                        intrinsics=args.intrinsics, 
                        database=args.database, 
                        output=args.mapper_path)
    
    logger.info(f"converted model saved to {args.mapper_path}")
        

    
  