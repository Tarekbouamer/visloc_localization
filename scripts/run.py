# General
from collections import OrderedDict
import argparse
from os import mkdir
import numpy as np
from pathlib import Path

# torch
import torch

# utils
from loc.utils.read_write_model import read_model
from loc.utils.configurations   import make_config
from loc.utils.logging          import setup_logger

# loc
from loc.dataset        import ImagesFromList, ImagesTransform
from loc.retrieve       import do_retrieve
from loc.matchers       import do_matching
from loc.localize       import main as localize
from loc.covisibility   import main as covisibility
from loc.triangulation  import main as triangulation

# colmap
from loc.colmap.colmap_nvm  import main as colmap_from_nvm
from loc.colmap.database    import COLMAPDatabase

# retrieval
import retrieval as ret

# configs
from loc.config import *

def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./cirtorch/configuration/defaults/global_config.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run")

    parser.add_argument('--resume', metavar='FILENAME', type=str,
                        help='name of the latest checkpoint (default: None)')

    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")


    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def init_device(args):
    """ Not used for the moment """
    # Initialize multi-processing
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = 0, 1

    # Set device
    torch.cuda.set_device(device_id)

    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def main(args):
    
    logger = setup_logger(output=".", name="loc")
    logger.info("init loc")
    
    #
    dataset_path    = Path('/media/dl/data_5tb/datasets/Vis2020/Aachen-Day-Night')
    outputs         = Path('/media/dl/data_5tb/datasets/Vis2020/Aachen-Day-Night/outputs')
    
    # dataset_path    = Path('/media/loc/data_5tb/datasets/Vis2020/Aachen-Day-Night/')
    # outputs         = Path('//media/loc/data_5tb/datasets/Vis2020/Aachen-Day-Night/outputs/')
    
    image_path      = dataset_path/'images/database_and_query_images/images_upright/' 
    
    sfm_pairs       = outputs / 'pairs-db-covis20.txt'    
    reference_sfm   = outputs / 'sfm_superpoint+superglue'  
    results         = outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  
    
    # outfolder
    if not outputs.exists():
        mkdir(outputs)
    
    # init extractor
    extractor = ret.FeatureExtractor(model_name='sfm_resnet50_gem_2048')
    
    # names
    meta = OrderedDict() 
    
    for split in ["query", "db"]:
        
        # data
        data_config = make_data_config(name='Aachen')
        
        #
        image_set = ImagesFromList(images_path=dataset_path/data_config[split]["images"], 
                                     split=split,
                                     max_size=600,
                                     logger=logger)
        
        #
        save_path = Path(str(outputs) + '/' + str(split) + '.h5')

        preds = extractor.extract_global(image_set, save_path=save_path)
        
        #
        print(preds["features"].shape)
        print(preds["save_path"])

        # # Meta
        # meta[split] = dict()
        # meta[split]["names"]    = image_set.get_names()
        # meta[split]["cameras"]  = image_set.get_cameras()

        # # Extract both, globals and locals  
        # features_path = feature_manager.extract(dataset=image_set,
        #                                         path=outputs/split,
        #                                         override=True,
        #                                         logger=logger) 
        
        # meta[split]["path"] = features_path
                
    
    # Nvm to colmap
    model_path = dataset_path / 'sfm_sift'

    colmap_from_nvm(dataset_path / '3D-models/aachen_cvpr2018_db.nvm',
                    dataset_path / '3D-models/database_intrinsics.txt',
                    dataset_path / 'aachen.db',
                    model_path, 
                    logger=logger,
                    override=True) 
    
    # Covisibility
    covisibility(model_path, sfm_pairs, num_matched=20, logger=logger)
        
    # Match SFM
    sfm_matches_path = outputs / Path('sfm_matches' +'.h5') 

    do_matching(src_path=Path(meta["db"]["path"] ), 
                dst_path=Path(meta["db"]["path"] ), 
                pairs_path=sfm_pairs, 
                output=sfm_matches_path, 
                override=True,
                logger=logger)
    
    # Triangulate
    reconstruction = triangulation(reference_sfm, model_path, image_path, sfm_pairs, meta["db"]["path"], sfm_matches_path,
                                   skip_geometric_verification=False, 
                                   estimate_two_view_geometries=False,
                                   verbose=True, logger=logger)
        
    # Retrieve
    num_top_matches = 20
    loc_pairs_path = outputs / Path('pairs' + '_' +  opts["retrieval_name"] + '_' +str(num_top_matches)  + '.txt') 
    do_retrieve(meta=meta, topK=num_top_matches, output=loc_pairs_path, override=False, logger=logger)
    meta["loc_pairs_path"] = str(loc_pairs_path)
    
    # Match
    loc_matches_path = outputs / Path('loc_matches_path' +'.h5') 

    do_matching(src_path=Path(meta["query"]["path"] ), dst_path=Path(meta["db"]["path"] ), 
                pairs_path=loc_pairs_path, 
                output=loc_matches_path, 
                logger=logger,
                override=False)
    
    # localize
    # localize(sfm_model=model_path,
    #          queries=meta["query"]["cameras"],
    #          retrieval_pairs_path=loc_pairs_path,
    #          features=Path(meta["query"]["path"] ),
    #          matches=loc_matches_path,
    #          results=results,
    #          covisibility_clustering=True,
    #          logger=logger,
    #          viewer=None
    #         )
    
    # Visualization
    # visualize_sfm_2d(model_path,  image_path,  n=4,    color_by='track_length'    )
    # visualize_sfm_2d(model_path,  image_path,  n=5,    color_by='visibility'      )
    # visualize_sfm_2d(model_path,  image_path,  n=5,    color_by='depth'           )
    # visualize_loc(results, image_path, model_path, n=5, top_k_db=2, prefix='query/night', seed=2)
    


    
    

    
    # if viewer3D is not None:
    #     viewer3D.draw_map()
    
    
if __name__ == '__main__':
  
    parser = make_parser()

    main(parser.parse_args())