import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.retrieval import image_retrieval
from loc.utils.logging import setup_logger


def run_image_retrieval_argparser():

    parser = argparse.ArgumentParser(
        description=('run image retrieval argparser'))

    parser.add_argument('--split',  type=str,  default="db", choices=["db", "query"],
                        help=' dataset split if any ')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')

    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def run_image_retrieval():

    # logger
    logger = setup_logger(output=".", name="loc")

    # args, cfg
    args = run_image_retrieval_argparser()
    default_cfg = OmegaConf.load(args.config)
    args, cfg = make_config(args, default_cfg)

    logger.info(f"run image retrieval")
    
    #
    image_retrieval(args, cfg)
   
if __name__ == '__main__':
    run_image_retrieval()
