import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

from loc.utils.viewer import VisualizerGui
from loc.utils.logging import setup_logger
from loc.tools.retrieval import do_retrieve
from loc.tools.matching import do_matching
from loc.tools.build_map_colmap import build_map_colmap
from loc.localize import do_localization
from loc.datasets.dataset import ImagesFromList
from loc.configurations.dataset_config import make_config
from omegaconf import OmegaConf
import logging
import argparse





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

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')
    return parser


def make_workspace(args, cfg):

    #
    logger = logging.getLogger("loc")

    # workspace
    args.workspace = Path(args.workspace)
    logger.info(f"workspace {args.workspace}")

    # images
    args.images_path = args.workspace / 'images'
    args.images_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"images {args.images_path}")

    # visloc
    args.visloc_path = args.workspace / 'visloc'
    args.visloc_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"visloc {args.visloc_path}")

    # mapper
    args.mapper_path = args.workspace / 'mapper'
    args.mapper_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"mapper {args.mapper_path}")

    return args


def main(args):

    # logger
    logger = setup_logger(output=".", name="loc")
    logger.info("init visloc_localization")

    # make config
    cfg = make_config(name='aachen', args=args)

    # make workspace
    args = make_workspace(args=args, cfg=cfg)

    # sfm pairs
    sfm_pairs_path = args.workspace / \
        str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
    sfm_matches_path = args.visloc_path / 'sfm_matches.h5'
    loc_matches_path = args.visloc_path / 'loc_matches.h5'

    # build map colmap
    mapper = build_map_colmap(
        args=args, cfg=cfg, sfm_pairs_path=sfm_pairs_path, sfm_matches_path=sfm_matches_path)

    # # convert 3d model
    # do_convert_3d_model(args=args, cfg=cfg)

    # # mapper
    # mapper = ColmapMapper(workspace=args.workspace, cfg=cfg)

    # # mapper.run_sfm()

    # # covisibility
    # sfm_pairs_path = mapper.covisible_pairs(sfm_pairs_path=sfm_pairs_path)

    # #
    # db_features_path, q_features_path = do_extraction(workspace=args.workspace,
    #                                                   save_path=args.visloc_path,
    #                                                   cfg=cfg)

    # # sfm pairs
    # sfm_matches_path = do_matching(src_path=db_features_path,
    #                                dst_path=db_features_path,
    #                                pairs_path=sfm_pairs_path,
    #                                cfg=cfg,
    #                                save_path=sfm_matches_path,
    #                                num_threads=args.num_threads)

    # # triangulate
    # reconstruction = do_reconstruction(mapper,
    #                                    sfm_pairs_path,
    #                                    db_features_path,
    #                                    sfm_matches_path)

    # retrieve
    loc_pairs_path = do_retrieve(workspace=args.workspace,
                                 save_path=args.visloc_path,
                                 cfg=cfg
                                 )

    # match
    loc_matches_path = do_matching(src_path=q_features_path,
                                   dst_path=db_features_path,
                                   pairs_path=loc_pairs_path,
                                   cfg=cfg,
                                   save_path=loc_matches_path,
                                   num_threads=args.num_threads)

    # localize
    # TODO: make sure datat config in cfg is fine for ImagesFrom List
    query_set = ImagesFromList(
        root=args.workspace, split="query", cfg=cfg, gray=True)

    do_localization(visloc_model=mapper.visloc_path,
                    queries=query_set.get_cameras(),
                    pairs_path=loc_pairs_path,
                    features_path=q_features_path,
                    matches_path=loc_matches_path,
                    cfg=cfg,
                    save_path=args.visloc_path)

    # vis_gui
    vis = VisualizerGui()
    vis.read_model(mapper.visloc_path)
    vis.create_window()
    vis.show()

    logger.info("Done")


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())
