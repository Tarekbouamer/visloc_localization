# logging
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loc.utils.writers import FeaturesWriter
# third
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint

from models.extractors import create_extractor

from .base import FeaturesExtractor

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
        self.model_name = model_name

        # self.extractor = create_model(model_name=model_name,
        #                               cfg=cfg)

        self.extractor = create_extractor(model_name=model_name,
                                      cfg=cfg.extractor)
                
        # init
        self._set_device()
        self._eval()

    def _unpack(self, preds):
        out = {}

        for k, v in preds.items():
            if isinstance(v, (List, Tuple)):
                v = v[0]
            out[k] = v

        return out

    @torch.no_grad()
    def extract_image(self,
                      data: Dict,
                      scales: List = [1.0],
                      **kwargs
                      ) -> dict:
        #
        original_size = data['size'][0]

        # prepare inputs
        data = self._prepare_input_data(data, **kwargs)

        # extract
        preds = self.extractor({'image': data["img"]})

        # unpack
        preds = self._unpack(preds)

        # scale keypoints to original scale
        current_size = data["img"].shape[-2:][::-1]
        scales = torch.Tensor(
            (original_size[0] / current_size[0], original_size[1] / current_size[1])).to(original_size).cuda()
        #
        preds['keypoints'] = (preds['keypoints'] + .5) * scales[None] - .5
        preds['uncertainty'] = preds.pop('uncertainty', 1.) * scales.mean()
        preds['size'] = original_size

        return preds

    @torch.no_grad()
    def extract_dataset(self,
                        dataset: Union[Dataset, DataLoader],
                        scales: List = [1.0],
                        save_path: Path = None,
                        **kwargs
                        ) -> Dict:
        # features writer
        self.writer = FeaturesWriter(save_path)

        # dataloader
        _dataloader = self._dataloader(dataset)

        # time
        start_time = time.time()

        # run -->
        for it, data in enumerate(tqdm(_dataloader, total=len(_dataloader), colour='green', desc='extract locals'.rjust(15))):
            #
            it_name = data['name'][0]

            #
            preds = self.extract_image(data, scales=scales, **kwargs)

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

        return save_path

    def __repr__(self) -> str:
        msg  = f"{self.__class__.__name__}"
        msg += f" ("
        msg += f" model_name: {self.model_name} "
        msg += f" device: {self.device} "
        msg += f" max_keypoints: {self.cfg.extractor.max_keypoints} "
        msg += f" max_size: {self.cfg.extractor.max_size} "
        msg += f" nms_radius: {self.cfg.extractor.nms_radius} "
        msg += f")"
        return msg