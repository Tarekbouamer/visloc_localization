# logger
import logging
from pathlib import Path

import numpy as np

from loc.datasets.dataset import ImagesFromList
from loc.extractors import GlobalExtractor, LocalExtractor

logger = logging.getLogger("loc")


def feature_extraction(workspace, split, save_path, cfg):

    #
    images = ImagesFromList(root=workspace, split=split, cfg=cfg)

    #
    loc_features_path = Path(str(save_path) + '/' +
                             str(split) + '_local_features.h5')
    glb_path_features = Path(str(save_path) + '/' +
                             str(split) + '_global_features.h5')

    # local
    local_extractor = LocalExtractor(cfg=cfg)

    logger.info(f" {local_extractor}")
    logger.info(f" local feature extractor to {loc_features_path}")

    local_extractor.extract_dataset(
        images, save_path=loc_features_path, normalize=False, gray=True)

    # global
    global_extractor = GlobalExtractor(cfg=cfg)

    logger.info(f" {global_extractor}")
    logger.info(f" global feature extractor to {glb_path_features}")

    global_extractor.extract_dataset(
        images, save_path=glb_path_features, normalize=True, gray=False)

    return save_path


def database_feature_extraction(workspace, save_path, cfg):
    return feature_extraction(workspace=workspace, split="db", save_path=save_path, cfg=cfg)


def query_feature_extraction(workspace, save_path, cfg):
    return feature_extraction(workspace=workspace, split="query", save_path=save_path, cfg=cfg)
