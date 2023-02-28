# logger
import logging
import os
from typing import Dict, List, Tuple

import gdown
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from features.register import is_model, model_entrypoint, register_detector

logger = logging.getLogger("loc")


def load_state_dict(checkpoint_path):
    """ load weights """

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        logger.info(f"Loaded from checkpoint {checkpoint_path}")
        return state_dict
    else:
        logger.error(f"No checkpoint found at {checkpoint_path}")
        raise FileNotFoundError()


def load_pretrained(model, variant, pretrained_cfg):

    #
    pretrained_file = pretrained_cfg.get('file',  None)
    pretrained_url = pretrained_cfg.get('url',   None)
    pretrained_drive = pretrained_cfg.get('drive', None)

    if pretrained_file:
        logger.info(
            f'Loading pretrained weights from file ({pretrained_file})')

        # load
        state_dict = load_state_dict(pretrained_file)

    elif pretrained_url:
        logger.info(f'Loading pretrained weights from url ({pretrained_url})')
        # load
        state_dict = load_state_dict_from_url(
            pretrained_url, map_location='cpu', progress=True, check_hash=False)

    elif pretrained_drive:
        #
        logger.info(
            f'Loading pretrained weights from google drive ({pretrained_drive})')

        #
        save_folder = "pretrained_drive"
        save_path = save_folder + "/" + variant + ".pth"

        # create fodler
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        # download from gdrive if weights not found
        if not os.path.exists(save_path):
            save_path = gdown.download(
                pretrained_drive, save_path, quiet=False, use_cookies=False)

        #  load from drive
        state_dict = load_state_dict(save_path)

    else:
        logger.warning(
            "No pretrained weights exist or were found for this model. Using random initialization.")
        return
    
    # load body and head weights
    strict = pretrained_cfg.get("strict", True)
    model.load_state_dict(state_dict, strict=strict)


def create_detector(detector_name: str,
                    cfg: Dict = {},
                    **kwargs
                    ) -> nn.Module:
    """create a detector from regitred models in detector factory

    Args:
        detector_name (str): name of detector in factory
        cfg (Dict, optional): configuration parameters. Defaults to None.

    Returns:
        nn.Module: detector model
    """

    #
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # check if detector exsists
    if not is_model(detector_name):
        raise RuntimeError('Unknown detector (%s)' % detector_name)

    #
    create_fn = model_entrypoint(detector_name)

    # create detector
    detector = create_fn(cfg=cfg, **kwargs)

    return detector
