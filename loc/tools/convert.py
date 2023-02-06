import logging
logger = logging.getLogger("loc")

from pathlib import Path

from loc.utils.colmap.colmap_nvm import main as colmap_from_nvm

def do_convert_3d_model(args, cfg):
    
    #
    logger.info(f"convert 3d model from {cfg.convert.type} to colmap format")

    # nvm -> colmap
    if cfg.convert.type == "nvm":
        colmap_from_nvm(nvm=Path(cfg.convert.nvm_path),
                        intrinsics=Path(cfg.convert.intrinsics),
                        database=Path(cfg.convert.database),
                        output=cfg.mapper_path) 
    
  