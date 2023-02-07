# logger
import logging

from tqdm import tqdm

from loc.datasets import PairsDataset
from loc.matchers import MatchSequence
from loc.utils.io import read_pairs_list, remove_duplicate_pairs

logger = logging.getLogger("loc")


def do_matching(pairs_path, src_path, dst_path, cfg=None, save_path=None, num_workers=4):
    """general matching 

    Args:
        pairs_path (str): pairs path
        src_path (str): src image features path
        dst_path (str): dst image features path
        save_path (str, optional): path to save matches. Defaults to None.
        num_workers (int, optional): number of workers. Defaults to 4.

    Returns:
        str: path to save matches 
    """

    # assert
    assert pairs_path.exists(), pairs_path
    assert src_path.exists(),   src_path
    assert dst_path.exists(),   dst_path

    # Load pairs
    pairs = read_pairs_list(pairs_path)
    pairs = remove_duplicate_pairs(pairs)

    if len(pairs) == 0:
        logger.error('no matches pairs found')
        return

    # pair dataset loader
    pair_dataset = PairsDataset(
        pairs=pairs, src_path=src_path, dst_path=dst_path)
    logger.info("matching %s pairs", len(pair_dataset))

    # matcher
    matcher = MatchSequence(cfg=cfg)
    save_path = matcher.match_sequence(pair_dataset, save_path)

    return save_path
