import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.triangulate import geometric_verification_and_triangulation
from loc.utils.logging import init_loguru


def triangulation_argparser():

    parser = argparse.ArgumentParser(
        description=('triangulation argparser'))

    parser.add_argument('--mode',  type=str,  default="sfm", choices=["sfm", "loc"],
                        help='exhaustive sfm matching, matches database images \
                        while loc between query and databse images ')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')

    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def run_triangulation():
    """geometric verification and triangulation 
    """    

    
    logger = init_loguru(name="loc", log_file="triangulation.log")

    # args
    args = triangulation_argparser()

    # cfg
    args, cfg = make_config(args)

    #
    logger.info(f"run geometric verification and triangulation")
    geometric_verification_and_triangulation(args, cfg)


if __name__ == '__main__':
    run_triangulation()
