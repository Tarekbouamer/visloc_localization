from loc.localize import do_localization
from loc.datasets.dataset import ImagesFromList

from loc.tools.matching import do_matching
from loc.tools.retrieval import do_retrieve, Retrieval  
    
# logger
import logging
logger = logging.getLogger("loc")

def run_localization(args, cfg, mapper):

    # 
    logger.info("compute retrieval")
    retrieval       = Retrieval(workspace=cfg.workspace, save_path=cfg.save_path, cfg=cfg)
    loc_pairs_path  = retrieval.retrieve()   

    # match
    loc_matches_path = do_matching(src_path=q_features_path,
                                   dst_path=db_features_path,
                                   pairs_path=loc_pairs_path,
                                   cfg=cfg,
                                   save_path=loc_matches_path,
                                   num_threads=cfg.num_threads)

    # localize
    query_set = ImagesFromList(
        root=cfg.workspace, split="query", cfg=cfg, gray=True)

    do_localization(visloc_model=mapper.visloc_path,
                    queries=query_set.get_cameras(),
                    pairs_path=loc_pairs_path,
                    features_path=q_features_path,
                    matches_path=loc_matches_path,
                    cfg=cfg,
                    save_path=cfg.visloc_path)