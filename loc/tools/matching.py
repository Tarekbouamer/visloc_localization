
from pathlib import Path
from typing import Dict

from loc.datasets import PairsDataset
from loc.matchers import MatchSequence
from loc.utils.io import read_pairs_list, remove_duplicate_pairs

from loguru import logger

def exhaustive_matching(pairs_path: Path, 
                        src_path: Path, 
                        dst_path: Path, 
                        cfg: Dict = None, 
                        save_path: Path = None
                        ) -> Path:
    """exhaustive matching:

        * read pairs
        * remove duplicates
        * compute matches

    Args:
        pairs_path (Path): image pairs path
        src_path (Path): source image features path
        dst_path (Path): destination image features path
        cfg (Dict, optional): configuration. Defaults to None.
        save_path (Path, optional): path to save matches. Defaults to None.

    Returns:
        Path: matches path
    """

    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # load pairs
    pairs = read_pairs_list(pairs_path)

    # remove duplicate pairs
    pairs = remove_duplicate_pairs(pairs)

    if len(pairs) == 0:
        logger.error('no matches pairs found')
        return

    # pair dataset loader
    pair_dataset = PairsDataset(
        pairs=pairs, src_path=src_path, dst_path=dst_path)
    logger.info(f"matching {len(pair_dataset)} pairs")

    # matcher
    matcher = MatchSequence(cfg=cfg)
    save_path = matcher.match_sequence(pair_dataset, save_path)

    return save_path
