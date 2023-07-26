# logger
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py

from loguru import logger

def path2key(name: str) -> str:
    """key id for item from its path

    Args:
        name str: path to item 

    Returns:
        str: key
    """
    return name.replace('/', '-')


def pairs2key(name0: str,
              name1: str,
              ) -> str:
    """key id of pairs items

    Args:
        name0 str: path to item0
        name1 str: path to item1

    Returns:
        str: key
    """
    separator = '/'
    return separator.join((path2key(name0), path2key(name1)))


def read_pairs_list(path: Union[str, Path]) -> List[Tuple]:
    """read pairs from txt file, and list them as tuples

    Args:
        path (Union[str, Path]): path to txt file 

    Returns:
        List[Tuple]: list of pairs (str, str)
    """
    pairs = []
    with open(path, 'r') as f:
        for line in f.read().rstrip('\n').split('\n'):

            if len(line) == 0:
                continue

            q_name, db_name = line.split()
            pairs.append((q_name, db_name))

    return pairs


def read_pairs_dict(path: Union[str, Path]) -> Dict[str,  List]:
    """read pairs from txt file in dictionary format

    Args:
        path (Union[str, Path]): path to txt file 

    Returns:
        Dict[str. List]: dict of pairs (str, List)
    """
    """
      Load retrieval pairs
    """
    pairs = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):

            if len(p) == 0:
                continue

            q_name, db_name = p.split()
            pairs[q_name].append(db_name)
    return pairs


def find_pair(hfile: h5py.File,
              name0: str,
              name1: str
              ) -> Tuple[str, bool]:
    """chekc of pairs key exsists in hfile for combinition (name0, name1) or its reverse (name1, name0)

    Args:
        hfile (h5py.File): _description_
        name0 (str): name item0
        name1 (str): name item1

    Raises:
        ValueError: if key not found inf hfile

    Returns:
        Tuple[str, bool]: return key and if reverse
    """
    
    if pairs2key(name0, name1) in hfile:
        return pairs2key(name0, name1), False

    if pairs2key(name1, name0) in hfile:
        return pairs2key(name1, name0), True

    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')


def remove_duplicate_pairs(pairs_all: List[Tuple[str]],
                           matches_path: Path = None
                           ) -> List[Tuple[str, str]]:
    """remove duplicate pairs F(name0, name1) == F(name1, name0)

    Args:
        pairs (List[Tuple[str]]): _description_
        matches_path (Path, optional): matches h5py file. Defaults to None.

    Returns:
        List[Tuple[str, str]]: filtred pairs
    """

    pairs = set()

    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    #
    pairs = list(pairs)

    if matches_path is not None and matches_path.exists():
        with h5py.File(str(matches_path), 'r', libver='latest') as hfile:
            pairs_filtered = []

            for i, j in pairs:
                if (pairs2key(i, j) in hfile or
                        pairs2key(j, i) in hfile):
                    continue
                pairs_filtered.append((i, j))

        return pairs_filtered

    return pairs
