import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from loc.configurations.dataset_config import make_config
from loc.tools.localization import dataset_localization
from loc.utils.logging import init_loguru


def run_dataset_localization_argparser():

    parser = argparse.ArgumentParser(
        description=('run dataset localization argparser'))

    parser.add_argument('--split',  type=str,  default="db", choices=["db", "query"],
                        help=' dataset split if any ')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')

    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def run_dataset_localization():
    """dataset localization
    """    

    
    logger = init_loguru(name="loc", log_file="dataset_localization.log")

    # args
    args = run_dataset_localization_argparser()
    
    # cfg
    args, cfg = make_config(args)

    logger.info(f"run dataset localization")
    
    #
    dataset_localization(args, cfg)
   
if __name__ == '__main__':
    run_dataset_localization()
