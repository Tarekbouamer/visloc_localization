# logger
import logging
import os
from typing import Dict, List, Tuple

import gdown
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from features.models.model_register import is_model, model_entrypoint

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


def load_pretrained(model, variant, pretrained_cfg, state_key=None, replace=None):

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
    
    print(state_dict.keys())
    # state key
    if state_key is not None:
        state_dict = state_dict[state_key]
        
    # replace
    if replace is not None:
        state_dict = {k.replace(replace[0], replace[1]): v for k, v in state_dict.items()}

    # load state
    model.load_state_dict(state_dict, strict=True)


def create_model(model_name: str,
                 cfg: Dict = {},
                 **kwargs
                 ) -> nn.Module:
    
    #
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # check if model exsists
    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    #
    create_fn = model_entrypoint(model_name)

    # create model
    model = create_fn(cfg=cfg, **kwargs)

    return model
