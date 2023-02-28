import argparse
import subprocess
from pathlib import Path

from loc.utils.io import run_command
from loc.utils.logging import setup_logger


def run_aachen_day_night_argparser():

    parser = argparse.ArgumentParser(
        description=('build aachen day night argparser'))

    parser.add_argument('--workspace',  type=str,
                        help='visloc folder format mapper, visloc, ... ')

    parser.add_argument('--config',  type=str,  default=Path("loc/configurations/aachen_day_night.yml"),
                        help='path to config file yml')

    return parser.parse_args()


def run_aachen_day_night(args):
    workspace = args.workspace
    cfg = args.config
    
    # convert model
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--dataset", "aachen",
    ]
    run_command(cmd="model_convert.py", args=args)
    
    # build map
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--dataset", "aachen",
        "--split", "db"
    ]
    run_command(cmd="build_visloc_map.py", args=args)

    # query extraction
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--dataset", "aachen",
        "--split", "query"
    ]
    run_command(cmd="feature_extraction.py", args=args)

    # retrieval
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--dataset", "aachen",
        "--split", "query"
    ]
    run_command(cmd="image_retrieval.py", args=args)

    # matching
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--mode", "loc",
    ]
    run_command(cmd="exhaustive_matching.py", args=args)

    # localization
    args = [
        "--workspace", workspace,
        "--config", cfg,
        "--dataset", "aachen",
        "--split", "query"
    ]
    run_command(cmd="dataset_localization.py", args=args)


if __name__ == '__main__':

    #
    logger = setup_logger(output=".", name="loc")

    #
    args = run_aachen_day_night_argparser()

    #
    run_aachen_day_night(args)
