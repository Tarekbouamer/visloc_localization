from typing import Any, Dict

from loguru import logger

from loc.mappers.colmap_mapper import ColmapMapper


def geometric_verification_and_triangulation(args: Any,
                                             cfg: Dict
                                             ) -> None:
    """geometric verification and triangulation

    Args:
        args (Any): arguments
        cfg (Dict): configuration
    """

    # paths
    db_features_path = args.visloc_path / 'db_local_features.h5'
    sfm_pairs_path = args.visloc_path / \
        str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
    sfm_matches_path = args.visloc_path / 'sfm_matches.h5'

    # mapper
    logger.info('init mapper')
    mapper = ColmapMapper(workspace=args.workspace, cfg=cfg)

    # create database
    mapper.create_database()

    # load features
    logger.info('import features to database')
    mapper.import_features(db_features_path)

    # load matches
    logger.info('import matches to database')
    mapper.import_matches(sfm_pairs_path, sfm_matches_path)

    # geometric verification
    logger.info('geometric verification')
    mapper.geometric_verification(
        db_features_path, sfm_pairs_path, sfm_matches_path)

    # triangulation
    logger.info('triangulation')
    mapper.triangulate_points(images_path=args.images_path,
                              verbose=True)

    return mapper
