# logger
import logging
from pathlib import Path

from tqdm import tqdm

from loc.datasets.dataset import ImagesFromList
from loc.localize import ImageLocalizer
from loc.matchers import Matcher
from loc.tools.retrieval import Retrieval, do_retrieve

logger = logging.getLogger("loc")


def run_localization(cfg, mapper):

    #
    loc_matches_path = Path(cfg.visloc_path) / 'loc_matches.h5'

    # retrieval
    retrieval = Retrieval(workspace=cfg.workspace,
                          save_path=cfg.visloc_path, cfg=cfg)

    # matcher
    matcher = Matcher(cfg=cfg)

    # localizer
    localizer = ImageLocalizer(
        visloc_model=cfg.visloc_path, retrieval=retrieval, matcher=matcher, cfg=cfg)

    # localize
    query_set = ImagesFromList(
        root=cfg.workspace, split="query", cfg=cfg, gray=True)

    for item in tqdm(query_set, total=len(query_set)):
        
        localizer(item)

        input()
