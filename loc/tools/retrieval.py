from typing import Dict, List, Tuple, Union, Any

import argparse
from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import torch
import collections.abc as collections
import os

from tqdm import tqdm

from loc.datasets.dataset import ImagesFromList
from loc.extractors import GlobalExtractor

from loc.utils.io import remove_duplicate_pairs
from loc.utils.readers import GlobalFeaturesLoader

from torch.utils.data import DataLoader

# logger
import logging
logger = logging.getLogger("loc")


class Retrieval(object):
    """general retrieval class

    Args:
        workspace (str): workspace path
        save_path (str): save path for global features
        cfg (dict): configurations
    """

    def __init__(self, workspace, save_path, cfg):

        # cfg
        self.cfg = cfg
        num_topK = self.cfg.retrieval.num_topK
        model_name = self.cfg.retrieval.model_name

        # extractor
        logger.info(f"init retrieval {model_name}")
        self.extractor = GlobalExtractor(cfg=cfg)

        #
        self.workspace = workspace
        self.save_path = save_path

        # paris path
        self.pairs_path = save_path / Path('pairs' + '_' +
                                           str(model_name) + '_' +
                                           str(num_topK) + '.txt')
        # preds
        self.db_features_path = Path(str(cfg.visloc_path) + '/' + 'db_features.h5')

        self.db_features_preds = None

    def _database_images_loader(self,):

        images_list = ImagesFromList(
            root=self.cfg.images_path, split="db", cfg=self.cfg)

        images_dl = DataLoader(
            images_list, num_workers=self.cfg.num_workers, drop_last=False)

        return images_dl

    def extract_database(self):
        """extract databse global features
        Returns:
            dict: database retrieval predictions         
        """
        split = "db"

        images = ImagesFromList(root=self.cfg.images_path, split=split, cfg=self.cfg)
        
        db_preds = self.extractor.extract_dataset(images, save_path=self.db_features_path, normalize=True, gray=False)

        self.db_features_preds = db_preds
        
        return db_preds

    def load_database_features(self):

        assert self.db_features_path.exists(), self.db_features_path

        logger.info(f"load global features from {self.db_features_path}")

        global_features_reader = GlobalFeaturesLoader(self.db_features_path)

        database_dl = self._database_images_loader()

        #
        features = []
        names = []

        for item in tqdm(database_dl):

            item_name = item["name"][0]
            
            print(item_name)

            pred = global_features_reader.load(item_name)

            features.append(pred)
            names.append(item_name)

        self.db_features_preds = {"features": torch.stack(features),
                                  "names": np.stack(names)
                                  }

    # def extract_images_queries(self):
    #     """extract query global features
    #     Returns:
    #         dict: query retrieval predictions         
    #     """
    #     logger.info(f"extract global features for query images ")
    #     db_preds = self.extract_images(self.workspace, split="query")
    #     return db_preds

    def _search(self, q_preds, db_preds):
        """descriptor bases matcher

        Args:
            q_preds (dict): query predictions 
            db_preds (dict): databse predictions

        Returns:
            dict: retrieval name pairs
        """
        #
        q_descs = q_preds["features"]
        db_descs = db_preds["features"]
        
        q_names = q_preds["names"]
        db_names = db_preds["names"]

        # similarity
        scores = torch.mm(q_descs, db_descs.t())

        # search for num_topK images
        num_topK = self.cfg.retrieval.num_topK

        invalid = np.array(q_names)[:, None] == np.array(db_names)[None]
        invalid = torch.from_numpy(invalid).to(scores.device)

        invalid |= scores < 0
        scores.masked_fill_(invalid, float('-inf'))

        topk = torch.topk(scores, k=num_topK, dim=1)

        indices = topk.indices.cpu().numpy()
        valid   = topk.values.isfinite().cpu().numpy()

        # collect pairs
        pairs = []
        for i, j in zip(*np.where(valid)):
            pairs.append((i, indices[i, j]))

        name_pairs = [(q_names[i], db_names[j]) for i, j in pairs]

        assert len(name_pairs) > 0, "No matching pairs has been found! "
        
        return name_pairs

    def _remove_duplicates(self, pairs):
        """remove duplicate pairs (a,b) (b,a)

        Args:
            pairs (dict): named pairs

        Returns:
            dict: filtred named pairs
        """
        return remove_duplicate_pairs(pairs)

    def retrieve(self, remove_duplicates=True):
        """retrieval main

        Args:
            remove_duplicates (bool, optional): remove duplicate pairs. Defaults to True.

        Returns:
            str: path to retrieval pairs (*.txt)
        """

        # extract
        db_preds = self.extract_database()
        q_preds = self.extract_images_queries()

        # match
        image_pairs = self._search(q_preds, db_preds)

        # remove duplicates
        if remove_duplicates:
            image_pairs = self._remove_duplicates(image_pairs)

        # save pairs
        with open(self.pairs_path, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in image_pairs))

        logger.info(
            f"{len(image_pairs)} pairs have been found, saved {self.pairs_path}")

        return self.pairs_path

    def __call__(self, item: Dict) -> Any:

        # database features
        db_preds = self.db_features_preds
        
        # extract query features
        q_preds = self.extractor.extract_image(item, normalize=True)
        
        #
        q_preds["names"] = np.array(item["name"])
        q_preds["features"] = q_preds["features"].unsqueeze(0)

        # search for pairs
        pairs = self._search(q_preds, db_preds)

        # 
        return pairs


def do_retrieve(workspace, save_path, cfg={}):
    """perform retrieval 

    Args:
        workspace (str): workspace path
        save_path (str): save path
        cfg (dict): configurations 

    Returns:
        str: path to retrieval pairs (*.txt)
    """

    return image_pairs
