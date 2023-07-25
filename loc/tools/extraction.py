# logger
import logging
from pathlib import Path
from typing import Dict

from loc.datasets.dataset import ImagesFromList
from loc.extractors import GlobalExtractor, LocalExtractor

logger = logging.getLogger("loc")


def feature_extraction(workspace: Path, 
                       split: str, 
                       save_path: Path, 
                       cfg: Dict
                       ) -> Path:
    """features extraction:
    
        * local features extraction
        * global features extraction

    Args:
        workspace (Path): workspace directory 
        split (str): dataset split
        save_path (Path): features save path
        cfg (Dict): configuration 

    Returns:
        Path: saved features path
    """    
    
    #
    images = ImagesFromList(root=workspace, split=split, cfg=cfg)

    #
    loc_features_path = Path(str(save_path) + '/' +
                             str(split) + '_local_features.h5')
    glb_path_features = Path(str(save_path) + '/' +
                             str(split) + '_global_features.h5')

    # local
    local_extractor = LocalExtractor(cfg=cfg)

    logger.info(f"{local_extractor}")
    logger.info(f"local feature extractor to {loc_features_path}")

    # local_extractor.extract_dataset(
    #     images, save_path=loc_features_path, normalize=False, gray=True)

    # global
    global_extractor = GlobalExtractor(cfg=cfg)

    logger.info(f"{global_extractor}")
    logger.info(f"global feature extractor to {glb_path_features}")

    global_extractor.extract_dataset(
        images, save_path=glb_path_features, normalize=True, gray=False)

    return save_path


def database_feature_extraction(workspace: Path,
                                save_path: Path,
                                cfg: Dict
                                ) -> Path:
    """database feature extraction

    Args:
        workspace (Path): path to workspace
        save_path (Path): path to save features
        cfg (Dict): configurations

    Returns:
        Path: save databse features
    """
    return feature_extraction(workspace=workspace, split="db", save_path=save_path, cfg=cfg)


def query_feature_extraction(workspace: Path,
                             save_path: Path,
                             cfg: Dict
                             ) -> Path:
    """query feature extraction

    Args:
        workspace (Path): path to workspace
        save_path (Path): path to save features
        cfg (Dict): configurations

    Returns:
        Path: saved query features
    """
    return feature_extraction(workspace=workspace, split="query", save_path=save_path, cfg=cfg)
