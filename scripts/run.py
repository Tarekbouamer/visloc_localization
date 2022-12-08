# General
import sys
from collections import OrderedDict
import argparse
from os import mkdir
import numpy as np
from pathlib import Path

# torch
import torch

# append
sys.path.append(str(Path(__file__).parent / '../'))

# utils
from loc.utils.read_write_model import read_model
from loc.utils.configurations   import make_config
from loc.utils.logging          import setup_logger

# loc
from loc.dataset        import ImagesFromList, ImagesTransform
from loc.retrieve       import do_retrieve
from loc.matching       import do_matching
from loc.localize       import main as localize
from loc.covisibility   import main as covisibility
from loc.triangulation  import main as triangulation
from loc.vis            import visualize_sfm_2d 

# colmap
from loc.colmap.colmap_nvm  import main as colmap_from_nvm
from loc.colmap.database    import COLMAPDatabase

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
    
    # dataset_path    = Path('/media/dl/data_5tb/datasets/Vis2020/Aachen-Day-Night')
    # outputs         = Path('/media/dl/data_5tb/datasets/Vis2020/Aachen-Day-Night/outputs')

    dataset_path    = Path('/media/loc/HDD/VisualLocalization2020/aachen/')
    outputs         = Path('/media/loc/HDD/VisualLocalization2020/aachen/visloc')    
    image_path      = dataset_path/'images/images_upright/' 

    # dataset_path    = Path('/media/loc/data_5tb/datasets/Vis2020/Aachen-Day-Night')
    # outputs         = Path('/media/loc/data_5tb/datasets/Vis2020/Aachen-Day-Night/outputs')    
    # image_path      = dataset_path/'images/database_and_query_images/images_upright/' 
    
    reference_sfm   = outputs / 'sfm_superpoint_mnn'  
    results         = outputs / 'Aachen_visloc_gem50.txt'  
    
    # outfolder
    if not outputs.exists():
        mkdir(outputs)
    
    # data config
    data_cfg = make_data_config(name='Aachen')
    
    # Nvm to Colmap
    model_path = outputs / 'sfm_sift'
    # colmap_from_nvm(dataset_path / '3D-models/aachen_cvpr2018_db.nvm',
    #                 dataset_path / '3D-models/database_intrinsics.txt',
    #                 dataset_path / 'aachen.db',
    #                 model_path) 
    
    # covisibility
    num_matched = 20
    sfm_pairs = outputs / str('sfm_pairs_' + str(num_matched) + '.txt') 
    # covisibility(model_path, sfm_pairs, num_matched=20)
    
    # locals
    logger.info("extract local features")

    #
    sp_cfg = {
        'keypoint_threshold': 0.005,    'remove_borders': 4,
        'nms_radius': 3,                'max_keypoints': 1024
        }

    # extract db images 
    detector    = SuperPoint(config=sp_cfg)
    db_set      = ImagesFromList(root=dataset_path, data_cfg=data_cfg, split='db',      max_size=600,   gray=True)
    db_path     = Path(str(outputs) + '/' + str('db') + '_local' + '.h5')
    # db_meta     = detector.extract_keypoints(db_set, save_path=db_path, normalize=False)        
  
    # extract query images 
    detector    = SuperPoint(config=sp_cfg)
    query_set   = ImagesFromList(root=dataset_path, data_cfg=data_cfg, split='query',   max_size=600,  gray=True)
    query_path  = Path(str(outputs) + '/' + str('query') + '_local' + '.h5')
    # query_meta  = detector.extract_keypoints(query_set, save_path=query_path, normalize=False)
       
    # sfm pairs
    sfm_matches_path = outputs / Path('sfm_matches' +'.h5') 
    # do_matching(src_path=db_path, 
    #             dst_path=db_path, 
    #             pairs_path=sfm_pairs, 
    #             output=sfm_matches_path)
    
    # triangulate
    reconstruction = triangulation(reference_sfm, 
                                   model_path, 
                                   image_path, 
                                   sfm_pairs, 
                                   db_path, 
                                   sfm_matches_path,
                                   skip_geometric_verification=False, verbose=True)
    
    # retrieve
    # loc_pairs_path = do_retrieve(dataset_path=dataset_path ,
    #                              data_cfg=data_cfg,
    #                              outputs=outputs,
    #                              topK=25) 
    
    # match
    loc_matches_path = outputs / Path('loc_matches_path' +'.h5') 
    # do_matching(src_path=query_path, 
    #             dst_path=db_path, 
    #             pairs_path=loc_pairs_path, 
    #             output=loc_matches_path)
    
    # localize
    # localize(sfm_model=model_path,
    #          queries=query_set.get_cameras(),
    #          retrieval_pairs_path=loc_pairs_path,
    #          features=query_path,
    #          matches=loc_matches_path,
    #          results=results,
    #          covisibility_clustering=True,
    #          viewer=None
    #         )
    
    # Visualization
    # visualize_sfm_2d(model_path,  image_path,  n=3,    color_by='track_length'    )
    # visualize_sfm_2d(model_path,  image_path,  n=3,    color_by='visibility'      )
    # visualize_sfm_2d(model_path,  image_path,  n=3,    color_by='depth'           )
    # visualize_loc(results, image_path, model_path, n=5, top_k_db=2, prefix='query/night', seed=2)
    
    # if viewer3D is not None:
    #     viewer3D.draw_map()
    
    
if __name__ == '__main__':
  
    parser = make_parser()

    main(parser.parse_args())