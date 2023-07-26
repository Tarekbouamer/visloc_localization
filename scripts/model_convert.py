import argparse
from pathlib import Path
from typing import Dict

from omegaconf import OmegaConf

from loc.configurations.dataset_config import make_config
from loc.tools.model_converter import run_model_conversion
from loc.utils.logging import init_loguru


def model_convert_argparser():

    parser = argparse.ArgumentParser(description=('model convert'))

    parser.add_argument('--type',  type=str,  default="nvm",
                        help='original model format')
    
    parser.add_argument('--model', type=str, help='path to original model')
    
    parser.add_argument('--intrinsics', type=str, 
                        help='path to camera intrinsics')
    
    parser.add_argument('--database', type=str, help='path to nvm database')
    
    parser.add_argument('--save_path', type=str,
                        help='path to save converted model')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')
    
    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def model_convert():
    # logger
    logger = init_loguru(name="loc", log_file="model_convert.log")

    #
    logger.info("run model convert")
    
    #
    args = model_convert_argparser()

    # make config
    args, cfg = make_config(args)
    
    # run
    run_model_conversion(args, cfg)


if __name__ == '__main__':
    model_convert()
