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


class Extraction(object):
    """local feature extractor

    Args:
        workspace (str): path to visloc workspace
        save_path (str): path to save directory
        cfg (str): configuration parameters
    """

    def __init__(self, workspace, save_path, cfg={}):

        # cfg
        self.cfg = cfg

        # extractor
        logger.info(f"init feature extractor {cfg.extractor.model_name}")

        self.extractor = LocalExtractor(cfg=cfg)

        #
        self.workspace = workspace
        self.save_path = save_path

    def extract_images(self, images_path, split=None):
        """extract local features

        Args:
            images_path (str): path to images
            split (str, optional): [query, db]. Defaults to None.

        Returns:
            dict: local predictions 
            str: local predictions path
        """

        images = ImagesFromList(
            root=images_path, split=split, cfg=self.cfg, gray=True)
        features_path = Path(str(self.save_path) + '/' +
                             str(split) + '_local_features.h5')
        logger.info(f"features will be saved to {features_path}")
        preds = self.extractor.extract_dataset(
            images, save_path=features_path, normalize=False)

        return preds, features_path

    def extract_images_database(self):
        """extract local features for database images

        Returns:
            dict: local database predictions 
            str: local database predictions path        
        """

        logger.info(f"extract local features for database images ")

        db_preds, db_path = self.extract_images(self.workspace, split="db")

        return db_preds, db_path

    def extract_images_queries(self):
        """extract local features for query images

        Returns:
            dict: local query predictions 
            str: local query predictions path  
        """

        logger.info(f"extract local features for query images ")

        q_preds, q_path = self.extract_images(self.workspace, split="query")

        return q_preds, q_path

    def extract(self):
        """local extraction main

        Returns:
            str: database local path
            str: query local path
        """
        # extract
        _, db_path = self.extract_images_database()
        _, q_path = self.extract_images_queries()

        return db_path, q_path


def database_feature_extraction(workspace, save_path, cfg):

    #
    split = "db"
    images = ImagesFromList(root=workspace, split=split, cfg=cfg)

    #
    db_local_features = Path(str(save_path) + '/' + 'db_local_features.h5')
    db_global_features = Path(str(save_path) + '/' + 'db_global_features.h5')

    # loca
    logger.info(
        f"local feature extractor {cfg.extractor.model_name} to {db_local_features}")

    local_extractor = LocalExtractor(cfg=cfg)

    loc_preds = local_extractor.extract_dataset(
        images, save_path=db_local_features, normalize=False, gray=True)

    # global
    logger.info(
        f"global feature extractor {cfg.retrieval.model_name} to {db_global_features}")

    global_extractor = GlobalExtractor(cfg=cfg)

    glb_preds = global_extractor.extract_dataset(
        images, save_path=db_global_features, normalize=True, gray=False)


    return save_path


def do_query_extraction(workspace, save_path, cfg):

    # extractor
    logger.info(f"init feature extractor {cfg.extractor.model_name}")

    extractor = LocalExtractor(cfg=cfg)

    split = "query"
    images = ImagesFromList(root=workspace, split=split, cfg=cfg, gray=True)
    features_path = Path(str(save_path) + '/' +
                         str(split) + '_features.h5')

    logger.info(f"features will be saved to {features_path}")
    preds = extractor.extract_dataset(
        images, save_path=features_path, normalize=False)

    return save_path


# def do_extraction(workspace, save_path, cfg):
#     """general extraction function
#     Args:
#         workspace (str): workspace path
#         save_path (str): save path
#         cfg (str): configurations
#     Returns:
#         str: path to retrieval pairs
#     """
#     ext = Extraction(workspace=workspace, save_path=save_path, cfg=cfg)
#     # run
#     db_path, q_path = ext.extract()
#     return db_path, q_path
