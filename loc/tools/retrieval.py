from pathlib import Path
from typing import Any, Dict

from loc.retrieval import Retrieval


def image_retrieval(args: Any,
                    cfg: Dict = None
                    ) -> Path:
    """image retrieval:

        * import features
        * search for closest images

    Args:
        args (Any): arguments 
        cfg (Dict, optional): configuration. Defaults to None.

    Returns:
        Path: retrieval pairs
    """

    # retrieval
    retrieval = Retrieval(workspace=args.workspace,
                          save_path=args.visloc_path,
                          cfg=cfg)
    # load features
    db_preds = retrieval.load_database_features(save_path=args.visloc_path)
    q_preds = retrieval.load_query_features(save_path=args.visloc_path)

    # search
    pairs_path = retrieval.retrieve(db_preds=db_preds,
                                    q_preds=q_preds)
    return pairs_path
