import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))

import argparse

from omegaconf import OmegaConf
from loc.configurations.dataset_config import make_config
from loc.tools.matching import exhaustive_matching
from loc.utils.logging import setup_logger



def exhaustive_matching_argparser():

    parser = argparse.ArgumentParser(
        description=('exhaustive matching argparser'))

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


def run_exhaustive_matching():

    # logger
    logger = setup_logger(output=".", name="loc")

    # args, cfg
    args = exhaustive_matching_argparser()
    default_cfg = OmegaConf.load(args.config)
    args, cfg = make_config(args, default_cfg)

    logger.info(f"run exhaustive matching {args.mode}")

    # features paths
    db_features_path = args.visloc_path / 'db_local_features.h5'
    query_features_path = args.visloc_path / 'query_local_features.h5'

    # matching mode
    
    if args.mode == "sfm":
        src_features_path = dst_features_path = db_features_path
        pairs_path = args.visloc_path / \
            str('sfm_pairs_' + str(cfg.mapper.num_covis) + '.txt')
        matches_path = args.visloc_path / 'sfm_matches.h5'
    
    elif args.mode == "loc":
        src_features_path = query_features_path
        dst_features_path = db_features_path
        pairs_path = args.visloc_path / str('loc_pairs'
                                            + '_' +
                                            str(cfg.retrieval.model_name)
                                            + '_' + str(cfg.retrieval.num_topK) + '.txt')

        matches_path = args.visloc_path / 'loc_matches.h5'
    else:
        raise KeyError(args.mode)

    #
    exhaustive_matching(pairs_path=pairs_path,
                        src_path=src_features_path,
                        dst_path=dst_features_path,
                        cfg=cfg,
                        save_path=matches_path)


if __name__ == '__main__':
    run_exhaustive_matching()
