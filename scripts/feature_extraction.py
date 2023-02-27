import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.extraction import database_feature_extraction, query_feature_extraction
from loc.utils.logging import setup_logger



def feature_extraction_argparser():

    parser = argparse.ArgumentParser(
        description=('feature extraction argparser'))

    parser.add_argument('--split',  type=str,  default="db", choices=["db", "query"],
                        help=' dataset split if any ')

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/default.yml"),
                        help='path to config file yml')

    parser.add_argument('--dataset',  type=str,  default="default",
                        help='dataset, if it exsists load defaults configuration parameters')

    return parser.parse_args()


def run_feature_extraction():
    """features extraction
    """    

    # logger
    logger = setup_logger(output=".", name="loc")

    # args
    args = feature_extraction_argparser()

    # cfg
    args, cfg = make_config(args)

    logger.info(f"run feature extraction {args.split}")
   
    # matching mode 
    if args.split == "db":
        database_feature_extraction(workspace=args.workspace, cfg=cfg, save_path=args.visloc_path)
    if args.split == "query":
        query_feature_extraction(workspace=args.workspace, cfg=cfg, save_path=args.visloc_path)
    else:
        raise KeyError(args.mode)


if __name__ == '__main__':
    run_feature_extraction()
