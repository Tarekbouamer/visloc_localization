import logging
from pathlib import Path

from loc.mappers.colmap_mapper import ColmapMapper
from loc.tools.convert import do_convert_3d_model
from loc.tools.extraction import database_feature_extraction
from loc.tools.matching import exhaustive_matching

# logger
logger = logging.getLogger("loc")


def build_map_colmap(cfg) -> None:
    """
    """

    # sfm paths
    sfm_pairs_path = Path(cfg.visloc_path) / \
        str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
    sfm_matches_path = Path(cfg.visloc_path) / 'sfm_matches.h5'

    # convert 3d model to colmap format
    logger.info('convert 3D model to colmap format')
    do_convert_3d_model(cfg=cfg)

    # mapper
    logger.info('init mapper')
    mapper = ColmapMapper(workspace=cfg.workspace, cfg=cfg)

    # covisibility
    logger.info('compute database covisibility pairs')
    sfm_pairs_path = mapper.covisible_pairs(sfm_pairs_path=sfm_pairs_path)

    # features extraction
    logger.info('extract database features')
    database_feature_extraction(workspace=cfg.workspace,
                                save_path=cfg.visloc_path,
                                cfg=cfg)

    db_features_path = Path(str(cfg.visloc_path) +
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
    reconstruction = mapper.triangulate_points(verbose=True)

    return mapper, db_features_path
