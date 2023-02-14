import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import os

from loc.datasets.dataset import ImagesFromList
from loc.extractors import LocalExtractor, GlobalExtractor


# logger
import logging
logger = logging.getLogger("loc")


def feature_extraction(workspace, split, save_path, cfg):

    #
    images = ImagesFromList(root=workspace, split=split, cfg=cfg)

    #
    loc_features_path = Path(str(save_path) + '/' +
                             str(split) + 'local_features.h5')
    glb_path_features = Path(str(save_path) + '/' +
                             str(split) + 'global_features.h5')

    # local
    local_extractor = LocalExtractor(cfg=cfg)

    logger.info(
        f"local feature extractor {local_extractor} to {loc_features_path}")

    local_extractor.extract_dataset(
        images, save_path=loc_features_path, normalize=False, gray=True)

    # global
    global_extractor = GlobalExtractor(cfg=cfg)

    logger.info(
        f"global feature extractor {global_extractor} to {glb_path_features}")

    global_extractor.extract_dataset(
        images, save_path=glb_path_features, normalize=True, gray=False)

    return save_path


def database_feature_extraction(workspace, save_path, cfg):
    return feature_extraction(workspace, save_path, cfg, split="db")
    
def query_feature_extraction(workspace, save_path, cfg):
    return feature_extraction(workspace, save_path, cfg, split="query")
