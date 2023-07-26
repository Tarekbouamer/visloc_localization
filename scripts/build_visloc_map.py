import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from loc.configurations.dataset_config import make_config
from loc.tools.build_visloc_map import build_visloc_map
from loc.utils.logging import init_loguru


def run_build_visloc_map_argparser():

    parser = argparse.ArgumentParser(
        description=('build visloc map argparser'))

    parser.add_argument('--split',  type=str,  default="db", choices=["db", "query"],
                        help=' dataset split if any ')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')

    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def run_build_visloc_map():
    """build a visloc map
    """    

    # logger
    logger = init_loguru(name="loc", log_file="build_visloc_map.log")

    # args
    args = run_build_visloc_map_argparser()
    
    args, cfg = make_config(args)

    logger.info("run build visloc map")
    
    #
    build_visloc_map(args, cfg)
   



if __name__ == '__main__':
    run_build_visloc_map()
