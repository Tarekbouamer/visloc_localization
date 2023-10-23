from pathlib import Path
from typing import Any, Dict

from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from loc.datasets.dataset import ImagesFromList
from loc.extractors import LocalExtractor
from loc.localize import DatasetLocalizer, ImageLocalizer
from loc.retrieval.retrieval import Retrieval
from loc.utils.io import dump_logs, write_poses_txt


def dataset_localization(args: Any,
                         cfg: Dict
                         ) -> None:
    """localize query set using extracted features.

    Args:
        args (Any): arguments
        cfg (Dict): configurations
    """

    #
    num_topK = cfg.retrieval.num_topK
    model_name = cfg.retrieval.model_name

    #
    features_path = args.visloc_path / 'query_local_features.h5'
    matches_path = args.visloc_path / 'loc_matches.h5'
    pairs_path = args.visloc_path / Path('loc_pairs' + '_' +
                                         str(model_name) + '_' +
                                         str(num_topK) + '.txt')

    #
    query_set = ImagesFromList(root=args.workspace, split="query", cfg=cfg)

    # localizer
    localizer = DatasetLocalizer(visloc_model=args.visloc_path,
                                 features_path=features_path,
                                 matches_path=matches_path,
                                 cfg=cfg)

    localizer(queries=query_set.get_cameras(),
              pairs_path=pairs_path,
              save_path=args.visloc_path)


def image_localization(args: Any,
                       cfg: Dict
                       ) -> None:
    """localize query set, extracting features and search for closest images from database

        * load database features
        * extract query features and compute camera pose

    Args:
        args (Any): arguments
        cfg (Dict): configurations
    """

    #
    db_features_path = args.visloc_path / 'db_local_features.h5'
    features_path = args.visloc_path / 'query_local_features.h5'

    # extractor
    extractor = LocalExtractor(cfg)
    logger.info(f"{extractor}")

    # retrieval
    retrieval = Retrieval(workspace=args.workspace,
                          save_path=args.visloc_path,
                          cfg=cfg)
    #
    retrieval.get_database_features()

    matches_path = args.visloc_path / 'loc_matches.h5'

    # localizer
    localizer = ImageLocalizer(visloc_model=args.visloc_path,
                               extractor=extractor,
                               retrieval=retrieval,
                               db_features_path=db_features_path,
                               features_path=features_path,
                               matches_path=matches_path,
                               cfg=cfg)

    # localize
    query_set = ImagesFromList(root=args.workspace, split="query", cfg=cfg)

    #
    query_dl = DataLoader(query_set, num_workers=cfg.num_workers)

    #
    logger.info('starting localization')

    for item in tqdm(query_dl, total=len(query_set)):

        # name
        qname = item['name'][0]

        if qname not in query_set.cameras:
            continue

        # camera params
        item["camera"] = query_set.cameras[qname]

        # localize
        localizer.localize_image(item)

    #
    poses = localizer.poses
    logs = localizer.logs

    #
    write_poses_txt(poses, args.visloc_path)
    dump_logs(logs, args.visloc_path)

    logger.info('done!')
