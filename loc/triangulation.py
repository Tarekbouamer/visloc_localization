import argparse
import contextlib
import io
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pycolmap

from loc.colmap.database import COLMAPDatabase

import h5py
from collections import defaultdict


def to_homogeneous(p):
    return np.pad(p, ((0, 0),) * (p.ndim - 1) + ((0, 1),), constant_values=1)


def pose_matrix_from_qvec_tvec(qvec, tvec):
    pose = np.zeros((4, 4))
    pose[: 3, : 3] = pycolmap.qvec_to_rotmat(qvec)
    pose[: 3, -1] = tvec
    pose[-1, -1] = 1
    return pose


def vector_to_cross_product_matrix(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def compute_epipolar_errors(qvec_r2t, tvec_r2t, p2d_r, p2d_t):
    T_r2t = pose_matrix_from_qvec_tvec(qvec_r2t, tvec_r2t)
    # Compute errors in normalized plane to avoid distortion.
    E = vector_to_cross_product_matrix(T_r2t[: 3, -1]) @ T_r2t[: 3, : 3]
    l2d_r2t = (E @ to_homogeneous(p2d_r).T).T
    l2d_t2r = (E.T @ to_homogeneous(p2d_t).T).T
    errors_r = (
        np.abs(np.sum(to_homogeneous(p2d_r) * l2d_t2r, axis=1)) /
        np.linalg.norm(l2d_t2r[:, : 2], axis=1))
    errors_t = (
        np.abs(np.sum(to_homogeneous(p2d_t) * l2d_r2t, axis=1)) /
        np.linalg.norm(l2d_r2t[:, : 2], axis=1))
    return E, errors_r, errors_t


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')


def find_pair(hfile: h5py.File, name0: str, name1: str):
    
    pair = names_to_pair(name0, name1)
    
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')
    
    
def get_matches(path, name0, name1):
    with h5py.File(str(path), 'r') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)

        matches = hfile[pair]['matches'].__array__()
        scores = hfile[pair]['scores'].__array__()
        
    #     print(matches)
    #     print(scores)
    #     print(matches.shape)
    #     print(scores.shape)   
    #     input()     
    # idx = np.where(matches != -1)
    # print(idx)   
    # print(idx[0].shape)   
    # print(idx[1].shape)   
   
    # matches = np.stack([idx, matches[idx]], -1)
    
    # if reverse:
    #     matches = np.flip(matches, -1)
    # scores = scores[idx]
    return matches, scores


def get_keypoints(path, name, return_uncertainty=False):
    
    with h5py.File(str(path), 'r') as hfile:
        dset = hfile[name]['keypoints']
        
        kpt         = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
        
    
    if return_uncertainty:
        return kpt, uncertainty
    
    return kpt


class OutputCapture:
    def __init__(self, verbose):
        self.verbose = verbose

    def __enter__(self):
        if not self.verbose:
            self.capture = contextlib.redirect_stdout(io.StringIO())
            self.out = self.capture.__enter__()

    def __exit__(self, exc_type, *args):
        if not self.verbose:
            self.capture.__exit__(exc_type, *args)
            if exc_type is not None:
                print('Failed with output:\n%s', self.out.getvalue())
        sys.stdout.flush()


def create_db_from_model(reconstruction, database_path, logger=None):
    if database_path.exists():
        
        if logger:
            logger.info('The database already exists, deleting it.')
            
        database_path.unlink()

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    for i, camera in reconstruction.cameras.items():
        db.add_camera(camera.model_id, camera.width, camera.height, camera.params, camera_id=i, prior_focal_length=True)

    for i, image in reconstruction.images.items():
        db.add_image(image.name, image.camera_id, image_id=i)

    db.commit()
    db.close()
    return {image.name: i for i, image in reconstruction.images.items()}


def import_features(image_ids, database_path, features_path, logger=None):
    
    if logger:
        logger.info('Importing features into the database...')
    
    db = COLMAPDatabase.connect(database_path)
        
    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path,
                   min_match_score=None, skip_geometric_verification=False, logger=None):
    
    if logger:
        logger.info('Importing matches into the database...')

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        
        matches, scores = get_matches(matches_path, name0, name1)
        
        if min_match_score:
            matches = matches[scores > min_match_score]
        
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}
        
        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()


def estimation_and_geometric_verification(database_path, pairs_path,
                                          verbose=False, logger=None):
    if logger:
        logger.info('Performing geometric verification of the matches...')
    
    with OutputCapture(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(database_path, pairs_path,max_num_trials=20000, min_inlier_ratio=0.1)


def geometric_verification(image_ids, reference, database_path, features_path,
                           pairs_path, matches_path, max_error=4.0, logger=None):

    if logger:
        logger.info('Performing geometric verification of the matches...')

    pairs = parse_retrieval(pairs_path)
    db = COLMAPDatabase.connect(database_path)

    inlier_ratios = []
    matched = set()
    for name0 in tqdm(pairs):
        
        id0     = image_ids[name0]
        
        image0  = reference.images[id0]
        cam0    = reference.cameras[image0.camera_id]
        
        kps0, noise0    = get_keypoints(features_path, name0, return_uncertainty=True)
        kps0            = np.array([cam0.image_to_world(kp) for kp in kps0])

        for name1 in pairs[name0]:
            id1 = image_ids[name1]
            image1 = reference.images[id1]
            cam1 = reference.cameras[image1.camera_id]
            
            kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
            kps1 = np.array([cam1.image_to_world(kp) for kp in kps1])

            matches = get_matches(matches_path, name0, name1)[0]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue
            
            matched |= {(id0, id1), (id1, id0)}

            if matches.shape[0] == 0:
                db.add_two_view_geometry(id0, id1, matches)
                continue
            
            qvec_01, tvec_01    = pycolmap.relative_pose(image0.qvec, image0.tvec, image1.qvec, image1.tvec)
            _, errors0, errors1 = compute_epipolar_errors(qvec_01, tvec_01, kps0[matches[:, 0]], kps1[matches[:, 1]])
            

            valid_matches = np.logical_and(
                errors0 <= max_error * noise0 / cam0.mean_focal_length(),
                errors1 <= max_error * noise1 / cam1.mean_focal_length())
            
            # TODO: We could also add E to the database, but we need
            # to reverse the transformations if id0 > id1 in utils/database.py.
            db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
            inlier_ratios.append(np.mean(valid_matches))
    
    if logger:
        logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                np.mean(inlier_ratios) * 100, np.median(inlier_ratios) * 100,
                np.min(inlier_ratios) * 100, np.max(inlier_ratios) * 100)

    db.commit()
    db.close()


def run_triangulation(model_path, database_path, image_dir, reference_model,
                      verbose=False, logger=None):
    model_path.mkdir(parents=True, exist_ok=True)
    
    if logger:
        logger.info('Running 3D triangulation...')
    
    with OutputCapture(verbose):
        with pycolmap.ostream():
            reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, image_dir, model_path)
    return reconstruction


def main(sfm_dir, model, image_dir, pairs, features, matches,
         skip_geometric_verification=False, estimate_two_view_geometries=False,
         min_match_score=None, verbose=False, logger=None):

    assert model.exists(),      model
    assert features.exists(),   features
    assert pairs.exists(),      pairs
    assert matches.exists(),    matches

    sfm_dir.mkdir(parents=True, exist_ok=True)
    
    database    = sfm_dir / 'database.db'
    reference   = pycolmap.Reconstruction(model)

    image_ids = create_db_from_model(reference, database, logger=logger)
    
    import_features(image_ids, database, features, logger=logger)
    
    import_matches(image_ids, database, pairs, matches, min_match_score, skip_geometric_verification, logger=logger)
    
    if not skip_geometric_verification:
        if estimate_two_view_geometries:
            estimation_and_geometric_verification(database, pairs, verbose, logger=logger)
        else:
            geometric_verification(image_ids, reference, database, features, pairs, matches, logger=logger)
    
    reconstruction = run_triangulation(sfm_dir, database, image_dir, reference,
                                       verbose, logger=logger)

    if logger:
        logger.info('Finished the triangulation with statistics:\n%s',
                reconstruction.summary())
    
    return reconstruction


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--reference_sfm_model', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)

    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)

    parser.add_argument('--skip_geometric_verification', action='store_true')
    parser.add_argument('--min_match_score', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(**args.__dict__)