# logger
import logging

import h5py
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("loc")

# TODO: work arround this


def read_key_from_h5py(name, _path, suffix=""):
    data = {}
    with h5py.File(str(_path), 'r', libver='latest') as f:

        if name in f:
            g = f[name]
        else:
            logger.error(f'{name} not found in {_path}')

        for k, v in g.items():
            data[k + str(suffix)] = torch.from_numpy(v.__array__()).float()

    return data


class PairsDataset(Dataset):
    """pair dataset

    Args:
        pairs (list): pairs list
        src_path (str): path to src image
        dst_path (str): path to dst image
    """

    def __init__(self, pairs, src_path, dst_path):

        self.pairs = pairs

        self.src_path = src_path
        self.dst_path = dst_path

    def __getitem__(self, idx):

        src_name, dst_name = self.pairs[idx]

        src = read_key_from_h5py(src_name, self.src_path, suffix="0")
        dst = read_key_from_h5py(dst_name, self.dst_path, suffix="1")

        data = {**src, **dst}

        return src_name, dst_name, data

    def __len__(self):
        return len(self.pairs)
