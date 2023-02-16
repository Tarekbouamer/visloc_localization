from pathlib import Path
import argparse

from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.gui import visualizer_gui
from loc.utils.logging import setup_logger

def visloc_gui_argparser():

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


def run_visloc_gui():

    # logger
    logger = setup_logger(output=".", name="loc")

    # args, cfg
    args = visloc_gui_argparser()
    default_cfg = OmegaConf.load(args.config)
    args, cfg = make_config(args, default_cfg)

    #
    logger.info(f"run visloc gui")
    visualizer_gui(args, cfg)


if __name__ == '__main__':
    run_visloc_gui()
