import logging
from pathlib import Path

from loc.datasets.dataset import ImagesFromList
from loc.mappers.colmap_mapper import ColmapMapper
from loc.tools.convert import do_convert_3d_model
from loc.tools.extraction import do_extraction
from loc.tools.matching import do_matching

# logger
logger = logging.getLogger("loc")


def build_map_colmap(args, cfg, sfm_pairs_path, sfm_matches_path) -> None:
    """
    """

    # convert 3d model
    logger.info('convert 3D model to colmap format')
    do_convert_3d_model(args=args, cfg=cfg)

    # mapper
    logger.info('init mapper')
    mapper = ColmapMapper(workspace=args.workspace, cfg=cfg)

    # covisibility
    logger.info('compute database covisibility pairs')
    sfm_pairs_path = mapper.covisible_pairs(sfm_pairs_path=sfm_pairs_path)

    #
    logger.info('extract database features')
    db_features_path, q_features_path = do_extraction(workspace=args.workspace,
                                                      save_path=args.visloc_path,
                                                      cfg=cfg)
    
    
    db_features_path  = Path(str(args.visloc_path) + '/' + str("db")    + '_local_features' + '.h5')
    q_features_path   = Path(str(args.visloc_path) + '/' + str("query") + '_local_features' + '.h5')

    logger.info('perform databse matching')
    sfm_matches_path = do_matching(src_path=db_features_path,
                                   dst_path=db_features_path,
                                   pairs_path=sfm_pairs_path,
                                   cfg=cfg,
                                   save_path=sfm_matches_path,
                                   num_threads=args.num_threads)
    
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
    reconstruction = mapper.triangulate_points()

    return mapper, db_features_path, q_features_path
