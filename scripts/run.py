from loc.utils.viewer import VisualizerGui
from loc.utils.logging import setup_logger
from loc.tools.run_localization import run_localization
from loc.tools.build_map_colmap import build_map_colmap
from loc.configurations.dataset_config import make_config
from omegaconf import OmegaConf
import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))


# from loc.mappers.colmap_mapper import ColmapMapper
# from loc.tools.extraction   import do_extraction
# from loc.tools.convert      import do_convert_3d_model
# from loc.tools.reconstruction import do_reconstruction


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
    cfg.workspace = cfg.workspace
    logger.info(f"workspace {cfg.workspace}")

    # images
    cfg.images_path = cfg.workspace + '/images'
    Path(cfg.images_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"images {cfg.images_path}")

    # visloc
    cfg.visloc_path = cfg.workspace + '/visloc'
    Path(cfg.visloc_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"visloc {cfg.visloc_path}")

    # mapper
    cfg.mapper_path = cfg.workspace + '/mapper'
    Path(cfg.mapper_path).mkdir(parents=True, exist_ok=True)
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

    # sfm pairs
    sfm_pairs_path = cfg.workspace / \
        str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
    sfm_matches_path = cfg.visloc_path / 'sfm_matches.h5'
    loc_matches_path = cfg.visloc_path / 'loc_matches.h5'

    # build map colmap
    mapper, db_features_path = build_map_colmap(
        cfg=cfg, sfm_pairs_path=sfm_pairs_path, sfm_matches_path=sfm_matches_path)

    run_localization(cfg=cfg, cfg=cfg, mapper=mapper)

    # vis_gui
    vis = VisualizerGui()
    vis.read_model(mapper.visloc_path)
    vis.create_window()
    vis.show()

    logger.info("Done")


if __name__ == '__main__':

    cli_cfg = OmegaConf.from_cli()
    main(cli_cfg)
