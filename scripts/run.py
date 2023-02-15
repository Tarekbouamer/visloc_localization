import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))
print(sys.path[-1])
input()
import logging
import argparse
from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.build_map_colmap import build_map_colmap
from loc.tools.run_localization import run_localization
from loc.utils.logging import setup_logger
from loc.utils.viewer import VisualizerGui


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='VisLoc Localization')

    parser.add_argument('--workspace', metavar='IMPORT_DIR',
                        help='data folder')
    parser.add_argument('--save_path', metavar='EXPORT_DIR',
                        help='path to localization folder')
    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='loc/configurations/default.yml')
    parser.add_argument('--num_threads', metavar='CST', type=int,
                        default=4, help='number of workers')

    cfg = parser.parse_cfg()

    for arg in vars(cfg):
        print(' {}\t  {}'.format(arg, getattr(cfg, arg)))

    print('\n ')
    return parser


def make_workspace(cfg):

    #
    logger = logging.getLogger("loc")

    # workspace
    cfg.workspace = Path(cfg.workspace)
    logger.info(f"workspace {cfg.workspace}")

    # images
    cfg.images_path = cfg.workspace / 'images'
    cfg.images_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"images {cfg.images_path}")

    # visloc
    cfg.visloc_path = cfg.workspace / 'visloc'
    cfg.visloc_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"visloc {cfg.visloc_path}")

    # mapper
    cfg.mapper_path = cfg.workspace / 'mapper'
    cfg.mapper_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"mapper {cfg.mapper_path}")

    return cfg


def main(cli_cfg):

    # logger
    logger = setup_logger(output=".", name="loc")
    logger.info("init visloc_localization")

    # make config
    cfg = make_config(name='aachen', cli_cfg=cli_cfg)

    # make workspace
    cfg = make_workspace(cfg=cfg)
    logger.info(OmegaConf.to_yaml(cfg))

    # build map colmap
    mapper, db_features_path = build_map_colmap(cfg=cfg)

    run_localization(cfg=cfg, mapper=mapper)

    # vis_gui
    vis = VisualizerGui()
    vis.read_model(mapper.visloc_path)
    vis.create_window()
    # vis.show()

    logger.info("Done")


if __name__ == '__main__':

    cli_cfg = OmegaConf.from_cli()
    main(cli_cfg)
