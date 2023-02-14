# logger
import logging
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from loc.datasets.dataset import ImagesFromList
from loc.extractors import LocalExtractor
from loc.localize import ImageLocalizer
from loc.matchers import MatchQueryDatabase
from loc.tools.retrieval import Retrieval

from loc.utils.io import (dump_logs,
                          write_poses_txt)

logger = logging.getLogger("loc")


def run_localization(cfg, mapper):

    # extractor
    extractor = LocalExtractor(cfg)
    
    # retrieval
    retrieval = Retrieval(workspace=cfg.workspace,
                          save_path=cfg.visloc_path, cfg=cfg)

    retrieval.load_database_features()

    # matcher
    matcher = MatchQueryDatabase(cfg=cfg)

    # localizer
    localizer = ImageLocalizer(
        visloc_model=mapper.visloc_path, extractor=extractor, retrieval=retrieval, matcher=matcher, cfg=cfg)

    # localize
    query_set = ImagesFromList(root=cfg.workspace, split="query", cfg=cfg)

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
        qpose = localizer.localize_image(item)

    #
    poses = localizer.poses
    logs = localizer.logs

    # 
    write_poses_txt(poses, cfg.visloc_path)
    dump_logs(logs, cfg.visloc_path)

    logger.info('done!')
