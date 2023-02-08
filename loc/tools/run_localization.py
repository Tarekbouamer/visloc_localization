# logger
import logging
from pathlib import Path

from torch.utils.data import DataLoader
from tqdm import tqdm

from loc.datasets.dataset import ImagesFromList
from loc.extractors import LocalExtractor
from loc.localize import ImageLocalizer
from loc.matchers import MatchQueryDatabase
from loc.tools.retrieval import Retrieval, do_retrieve

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
        visloc_model=cfg.visloc_path, extractor=extractor, retrieval=retrieval, matcher=matcher, cfg=cfg)

    # localize
    query_set = ImagesFromList(
        root=cfg.workspace, split="query", cfg=cfg, gray=False)

    #
    query_dl = DataLoader(query_set, num_workers=cfg.num_workers)

    #
    for item in tqdm(query_dl, total=len(query_set)):
        
        # camera
        item["camera"] = query_set.cameras[item['name'][0]]
        
        # localize
        localizer(item)

        input()
