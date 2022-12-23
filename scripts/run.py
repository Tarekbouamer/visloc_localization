# General
import sys
from collections import OrderedDict
import argparse
from os import mkdir
import numpy as np
from pathlib import Path
import time
# 
from omegaconf import OmegaConf

# torch
import torch

# append
sys.path.append(str(Path(__file__).parent / '../'))

# utils
from loc.utils.colmap.read_write_model import read_model
from loc.utils.configurations   import make_config
from loc.utils.logging          import setup_logger

# loc
from loc.dataset        import ImagesFromList, ImagesTransform

from loc.extract        import do_extraction
from loc.retrieve       import do_retrieve
from loc.matching       import do_matching
from loc.localize       import main as localize
from loc.covisibility   import main as covisibility
from loc.triangulation  import main as triangulation
from loc.vis            import visualize_sfm_2d 

from loc.utils.viewer3d import Model

# colmap
from loc.utils.colmap.colmap_nvm  import main as colmap_from_nvm
from loc.utils.colmap.database    import COLMAPDatabase

# mapper
from loc.mappers.colmap_mapper  import ColmapMapper

# retrieval
# import retrieval as ret

# detectors
from loc.extractors      import SuperPoint, FeatureExtractor

# third 

# configs
from loc.config import *

def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    parser.add_argument('--directory', metavar='IMPORT_DIR',
                        help='data folder')
    parser.add_argument('--save_path', metavar='EXPORT_DIR',
                        help='path to localization folder')
    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='loc/configurations/default.yml')

    args = parser.parse_args()
    
    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')
    return parser


def main(args):
 

    
    # logger
    logger = setup_logger(output=".", name="loc")
    logger.info("init visloc_localization")
    
    args.directory = Path(args.directory)
    logger.info(f"data workspace {args.directory}")
    
    if args.save_path is None:
        args.save_path = args.directory / 'visloc' 
    
    logger.info(f"visloc workspace {args.save_path}")
    args.save_path.mkdir(parents=True, exist_ok=True)
        
    # load config 
    cfg = OmegaConf.load(args.config)

    # data config
    data_cfg = make_data_config(name='aachen')
    
    # merge
    cfg = OmegaConf.merge(cfg, data_cfg)
    
    # #TODO: Nvm to Colmap
    # colmap_from_nvm(dataset_path / '3D-models/aachen_cvpr2018_db.nvm',
    #                 dataset_path / '3D-models/database_intrinsics.txt',
    #                 dataset_path / 'aachen.db',
    #                 model_path) 
    
    
    # mapper  
    mapper = ColmapMapper(data_path=args.directory,
                          workspace=args.save_path, 
                          cfg=cfg)
    
    # # covisibility
    # sfm_pairs_path = mapper.covisible_pairs()

    # # 
    # db_features_path, q_features_path = do_extraction(workspace=args.directory,
    #                                                   save_path=args.save_path,
    #                                                   cfg=cfg)

    # # sfm pairs
    # sfm_matches_path = args.save_path / 'sfm_matches.h5' 
    # sfm_matches_path = do_matching( src_path=db_features_path, 
    #                                 dst_path=db_features_path, 
    #                                 pairs_path=sfm_pairs_path, 
    #                                 save_path=sfm_matches_path)
    
    # # triangulate
    # reconstruction = triangulation(mapper, 
    #                                sfm_pairs_path, 
    #                                db_features_path, 
    #                                sfm_matches_path)
    
    # # # retrieve
    # loc_pairs_path = do_retrieve(workspace=args.directory ,
    #                              save_path=args.save_path,
    #                              cfg=cfg
    #                              ) 
    
    # # match
    # loc_matches_path = args.save_path / 'loc_matches.h5' 
    # loc_matches_path= do_matching(src_path=q_features_path, 
    #                               dst_path=db_features_path, 
    #                               pairs_path=loc_pairs_path, 
    #                               save_path=loc_matches_path)
    
    # # localize
    # query_set = ImagesFromList(root=args.directory, split="query", cfg=data_cfg, gray=True)

    # localize(sfm_model=mapper.visloc_model_path,
    #          queries=query_set.get_cameras(),
    #          pairs_path=loc_pairs_path,
    #          features=q_features_path,
    #          matches=loc_matches_path,
    #          results=args.save_path)
    import pycolmap
    sfm_model = pycolmap.Reconstruction(mapper.visloc_model_path)  
    model = Model()
    model.read_model(mapper.colmap_model_path)
    model.create_window()
    # model.add_points(min_track_len=4)
    model.add_cameras(scale=0.25)
    model.show()

    # 
    # viewer = Viewer3D()   
    # viewer.draw_sfm(mapper.visloc_model_path)

    # while(True):
    #     time.sleep(10)
    
if __name__ == '__main__':
  
    parser = make_parser()

    main(parser.parse_args())