# General
import argparse
import sys
import time
from collections import OrderedDict
from os import mkdir
from pathlib import Path

import numpy as np
# torch
import torch
# 
from omegaconf import OmegaConf

# append
sys.path.append(str(Path(__file__).parent / '../'))

# configs
from loc.configurations.dataset_config import *
# loc
from loc.datasets.dataset import ImagesFromList, ImagesTransform
# detectors
from loc.extractors import FeatureExtractor, SuperPoint
from loc.localize import do_localization
# mapper
from loc.mappers.colmap_mapper import ColmapMapper
from loc.tools.extract import do_extraction
from loc.tools.matching import do_matching
from loc.tools.reconstruction import do_reconstruction
from loc.tools.retrieve import do_retrieve
# colmap
from loc.utils.colmap.colmap_nvm import main as colmap_from_nvm
from loc.utils.colmap.database import COLMAPDatabase
# utils
from loc.utils.colmap.read_write_model import read_model
from loc.utils.configurations import make_config
from loc.utils.logging import setup_logger
from loc.utils.viewer import VisualizerGui, Visualizer

# retrieval
# import retrieval as ret


# third 


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='VisLoc Localization')

    parser.add_argument('--workspace', metavar='IMPORT_DIR',
                        help='data folder')
    parser.add_argument('--save_path', metavar='EXPORT_DIR',
                        help='path to localization folder')
    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='loc/configurations/default.yml')
    parser.add_argument('--num_threads', metavar='CST',type=int,
                        default=4, help='number of workers')

    args = parser.parse_args()
    
    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')
    return parser


def main(args):
 
    # logger
    logger = setup_logger(output=".", name="loc")
    logger.info("init visloc_localization")
    
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
        
    # load config 
    cfg = OmegaConf.load(args.config)
    
    # data config
    data_cfg = make_data_config(name='aachen')
    
    # merge
    cfg = OmegaConf.merge(cfg, data_cfg)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # #TODO: Nvm to Colmap
    # colmap_from_nvm(dataset_path / '3D-models/aachen_cvpr2018_db.nvm',
    #                 dataset_path / '3D-models/database_intrinsics.txt',
    #                 dataset_path / 'aachen.db',
    #                 model_path) 
    
    
    # mapper  
    mapper = ColmapMapper(workspace=args.workspace, 
                          images_path=args.images_path,
                          cfg=cfg)

    mapper.run_sfm()
    
    # # covisibility
    sfm_pairs_path = mapper.covisible_pairs()

    # # 
    db_features_path, q_features_path = do_extraction(workspace=args.workspace,
                                                      save_path=args.save_path,
                                                      cfg=cfg)

    # # sfm pairs
    sfm_matches_path = args.save_path / 'sfm_matches.h5' 
    sfm_matches_path = do_matching( src_path=db_features_path, 
                                    dst_path=db_features_path, 
                                    pairs_path=sfm_pairs_path, 
                                    save_path=sfm_matches_path,
                                    num_threads=args.num_threads)

    
    # # triangulate
    reconstruction = do_reconstruction(mapper, 
                                   sfm_pairs_path, 
                                   db_features_path, 
                                   sfm_matches_path)
    
    # # # retrieve
    loc_pairs_path = do_retrieve(workspace=args.workspace ,
                                 save_path=args.save_path,
                                 cfg=cfg
                                 ) 
    
    # # match
    loc_matches_path = args.save_path / 'loc_matches.h5' 
    loc_matches_path= do_matching(src_path=q_features_path, 
                                  dst_path=db_features_path, 
                                  pairs_path=loc_pairs_path, 
                                  save_path=loc_matches_path,
                                  num_threads=args.num_threads)
    
    # # localize
    query_set = ImagesFromList(root=args.workspace, split="query", cfg=data_cfg, gray=True)

    do_localization(visloc_model=mapper.visloc_model_path,
                    queries=query_set.get_cameras(),
                    pairs_path=loc_pairs_path,
                    features=q_features_path,
                    matches=loc_matches_path,
                    cfg=cfg,
                    save_path=args.save_path)

    # vis_gui
    vis = VisualizerGui()
    vis.read_model(mapper.visloc_model_path)
    vis.create_window()
    vis.show()

    
if __name__ == '__main__':
  
    parser = make_parser()

    main(parser.parse_args())