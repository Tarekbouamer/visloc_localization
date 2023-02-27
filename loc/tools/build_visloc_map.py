import logging
from typing import Dict, Any
from pathlib import Path

from loc.mappers.colmap_mapper import ColmapMapper
from loc.tools.extraction import database_feature_extraction
from loc.tools.matching import exhaustive_matching

import pycolmap

logger = logging.getLogger("loc")


def build_visloc_map(args: Any, 
                     cfg: Dict
                     ) -> pycolmap.Reconstruction:
    
    """build a visloc map by running :
    
        * compute covisibility sfm pairs
        * extract database features
        * exhaustive matching
        * make colmap databse, import features and matches
        * geometric verification
        * triangulation

    Args:
        args (Any): arguments
        cfg (Dict): configuration 

    Returns:
        pycolmap.Reconstruction: model reconstruction
    """

    # sfm paths
    sfm_pairs_path = Path(args.visloc_path) / \
        str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
    sfm_matches_path = Path(args.visloc_path) / 'sfm_matches.h5'

    # mapper
    logger.info('init mapper')
    mapper = ColmapMapper(workspace=args.workspace, cfg=cfg)

    # covisibility
    logger.info('compute database covisibility pairs')
    sfm_pairs_path = mapper.covisible_pairs(sfm_pairs_path=sfm_pairs_path)

    # features extraction
    logger.info('extract database features')
    database_feature_extraction(workspace=args.workspace,
                                save_path=args.visloc_path,
                                cfg=cfg)

    db_features_path = Path(str(args.visloc_path) +
                            '/' + 'db_local_features.h5')

    # exhaustive matching
    logger.info('perform databse matching')
    sfm_matches_path = exhaustive_matching(src_path=db_features_path,
                                           dst_path=db_features_path,
                                           pairs_path=sfm_pairs_path,
                                           cfg=cfg,
                                           save_path=sfm_matches_path)

    # make colmap database
    mapper.create_database()

    logger.info('import features to database')
    mapper.import_features(db_features_path)

    logger.info('import matches to database')
    mapper.import_matches(sfm_pairs_path, sfm_matches_path)

    logger.info('geometric verification')
    mapper.geometric_verification(
        db_features_path, sfm_pairs_path, sfm_matches_path)

    # triangulate
    logger.info('triangulation')
    model = mapper.triangulate_points(images_path=args.images_path,
                                      verbose=True)

    return model
