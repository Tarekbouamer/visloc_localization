from typing import Any, Union, Dict, List, Tuple

from pathlib import Path

import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from .base import FeaturesExtractor
from retrieval.datasets import ImagesListDataset
from retrieval.models import create_model, get_pretrained_cfg
from retrieval.utils.logging import setup_logger

from loc.utils.writers import FeaturesWriter

# third
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

# logging
import logging
logger = logging.getLogger("loc")


def create_model(model_name, cfg):
    if model_name == "superpoint":
        return SuperPoint(config=cfg.extractor)
    else:
        raise KeyError


class LocalExtractor(FeaturesExtractor):
    def __init__(self,
                 cfg: Dict = None
                 ) -> None:
        super().__init__(cfg)

        #
        model_name = self.cfg.extractor.model_name
        self.extractor = create_model(model_name=model_name,
                                      cfg=cfg)

        # init
        self._set_device()
        self._eval()

    def _to_numpy(self, preds):
        out = {}

        for k, v in preds.items():

            # ? list
            if isinstance(v, (List, Tuple)):
                v = v[0]

            # numpy
            if v.is_cuda:
                out[k] = v.cpu().numpy()
            else:
                out[k] = v.numpy()
        return out

    @torch.no_grad()
    def extract_dataset(self,
                        dataset: Union[Dataset, DataLoader],
                        scales: List = [1.0],
                        save_path: Path = None,
                        normalize: bool = False
                        ) -> Dict:

        # features writer
        self.writer = FeaturesWriter(save_path)

        # dataloader
        _dataloader = self._dataloader(dataset)

        # time
        start_time = time.time()

        # run -->
        for it, data in enumerate(tqdm(_dataloader, total=len(_dataloader), colour='green', desc='extract global'.rjust(15))):

            #
            it_name = data['name'][0]
            original_size = data['size'][0].numpy()

            # prepare inputs
            data = self._prepare_inputs(data)

            # extract locals
            preds = self.extractor({'image': data["img"]})
            preds = self._to_numpy(preds)

            # scale keypoints to original scale
            current_size = np.array(data["img"].shape[-2:][::-1])
            scales = (original_size / current_size).astype(np.float32)

            #
            preds['keypoints'] = (preds['keypoints'] + .5) * scales[None] - .5
            preds['uncertainty'] = preds.pop('uncertainty', 1.) * scales.mean()
            preds['size'] = original_size

            # write preds
            self.writer.write_items(key=it_name, data=preds)

            # clear cache
            if it % 10 == 0:
                torch.cuda.empty_cache()

        # close writer
        self.writer.close()

        # end time
        end_time = time.time() - start_time

        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')

        #
        out = {
            "save_path":  save_path
        }

        return out