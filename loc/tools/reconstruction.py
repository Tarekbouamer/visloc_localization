# logger
import logging
logger = logging.getLogger("loc")


def do_reconstruction(mapper, sfm_pairs_path, db_features_path, sfm_matches_path):
    """general reconstction function 

    Args:
        mapper (Mapper): _description_
        sfm_pairs_path (str): path to image pairs
        db_features_path (str): path to features data (*.h5)
        sfm_matches_path (str): path to matches data (*.h5)

    Returns:
        pycolmap.Reconstruction: reconstruction 
    """    
 
    assert db_features_path.exists(),   db_features_path
    assert sfm_pairs_path.exists(),     sfm_pairs_path
    assert sfm_matches_path.exists(),   sfm_matches_path
    
    # 
    mapper.create_database()
    mapper.import_features(db_features_path)
    mapper.import_matches(sfm_pairs_path, sfm_matches_path)

    #
    mapper.geometric_verification(db_features_path, sfm_pairs_path, sfm_matches_path)

    #
    reconstruction = mapper.triangulate_points()

    return reconstruction
