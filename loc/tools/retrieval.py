import logging

from loc.retrieval import Retrieval

logger = logging.getLogger("loc")


def image_retrieval(args, cfg=None):

    # retrieval
    retrieval = Retrieval(workspace=args.workspace,
                          save_path=args.visloc_path,
                          cfg=cfg)
    # load features
    db_preds = retrieval.load_database_features(save_path=args.visloc_path)
    q_preds = retrieval.load_query_features(save_path=args.visloc_path)

    # 
    pairs_path = retrieval.retrieve(db_preds=db_preds,
                                    q_preds=q_preds)
    return pairs_path
