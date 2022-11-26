import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import pickle
import pycolmap

from typing import Dict, List, Union, Tuple
from collections import defaultdict

# from loc.vis import visualize_localization_3d
from loc.utils.io import parse_retrieval_file

def get_keypoints(path: Path, name: str,
                  return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), 'r') as hfile:
        dset = hfile[name]['keypoints']
        p = dset.__array__()
        uncertainty = dset.attrs.get('uncertainty')
    if return_uncertainty:
        return p, uncertainty
    return p
  

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
    
    
def get_matches(path: Path, name0: str, name1: str) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r') as hfile:

        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches'].__array__()
        scores  = hfile[pair]['scores'].__array__()
        
    # idx = np.where(matches != -1)[0]
    # matches = np.stack([idx, matches[idx]], -1)
    # if reverse:
    #     matches = np.flip(matches, -1)
    # scores = scores[idx]
    
    return matches, scores

  
def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')


def do_covisibility_clustering(frame_ids: List[int], sfm_model: pycolmap.Reconstruction):
    """
        cluster database U retrieved images using covisility graph.
    """
    
    clusters    = []
    visited     = set()
   
    for frame_id in frame_ids:
        
        # Check if already labeled
        if frame_id in visited:
            continue
                
        # New component
        clusters.append([])
        queue = {frame_id}

        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            # Covisible frames 
            observed = sfm_model.images[exploration_frame].points2D
            
            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D() for obs in sfm_model.points3D[p2D.point3D_id].track.elements
            }
            
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    # 
    clusters = sorted(clusters, key=len, reverse=True)
    
    return clusters


class Localizer:
    def __init__(self, sfm_model, config=None):
        self.sfm_model = sfm_model
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.sfm_model.points3D[j].xyz for j in points3D_id]
        
        # EPnP pose estimation
        ret = pycolmap.absolute_pose_estimation(points2D, 
                                                points3D,
                                                query_camera,
                                                estimation_options=self.config.get('estimation', {}),
                                                refinement_options=self.config.get('refinement', {}),
        )
        return ret


def pose_from_cluster(
        localizer: Localizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        features_path: Path,
        matches_path: Path,
        logger=None,
        **kwargs):
    
    # Get Query 2D keypoints
    kpq = get_keypoints(features_path, qname)
    kpq += 0.5  # COLMAP coordinates

    # Get Visible 3D points
    kp_idx_to_3D        = defaultdict(list)
    kp_idx_to_3D_to_db  = defaultdict(lambda: defaultdict(list))
    num_matches = 0
    
    # 
    for i, db_id in enumerate(db_ids):
        
        image = localizer.sfm_model.images[db_id]
        
        if image.num_points3D() == 0:
            logger.debug(f'No 3D points found for {image.name}.')
            continue
        
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])
        
        matches, _ = get_matches(matches_path, qname, image.name)
        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            # avoid duplicate observations
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    
    # Localize 
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
        
    ret['camera'] = {
        'model': query_camera.model_name,
        'width': query_camera.width,
        'height': query_camera.height,
        'params': query_camera.params,
    }

    # mostly for logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': db_ids,
        'PnP_ret': ret,
        'keypoints_query': kpq[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,  # we don't log xyz anymore because of file size
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }
    return ret, log


def main(sfm_model: Union[Path, pycolmap.Reconstruction],
         queries: dict,
         retrieval_pairs_path: Path,
         features: Path,
         matches: Path,
         results: Path,
         ransac_thresh: int = 12,
         covisibility_clustering: bool = False,
         prepend_camera_name: bool = False,
         config: Dict = None, 
         logger=None,
         viewer=None):

    assert retrieval_pairs_path.exists(),  retrieval_pairs_path
    
    assert features.exists(),   features
    assert matches.exists(),    matches
    
    # Load retrieval pairs
    retrievals = parse_retrieval_file(retrieval_pairs_path)

    logger.info('Reading the 3D model...') 
    if not isinstance(sfm_model, pycolmap.Reconstruction):
        sfm_model = pycolmap.Reconstruction(sfm_model)
        
    db_name_to_id = {img.name: i for i, img in sfm_model.images.items()}

    config = {"estimation": {"ransac": {"max_error": ransac_thresh}},
              **(config or {})}
    
    # Localizer
    localizer = Localizer(sfm_model, config)

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrievals,
        'loc': {},
    }
    
    
    logger.info('Starting localization...')
    
    for qname, qcam in tqdm(queries.items(), total=len(queries)):
        
        #  
        if qname not in retrievals:
            logger.debug(f'No images retrieved for query image {qname}. Skipping...')
            continue
        
        # Geo-Verification
        db_names = retrievals[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.debug(f'Image {n} was retrieved but not in database')
                continue
                
            db_ids.append(db_name_to_id[n])
        
        #      
        if len(db_ids)<1:
            logger.error(" Empty retrieval")
            exit(0)
        
        # Covisibility Clustering --> Hloc
        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, sfm_model)
            
            best_inliers    = 0
            best_cluster_id = None
            logs_clusters   = []
            
            # For each cluser compute the query pose
            for id, cluster_ids in enumerate(clusters):
                
                ret, log = pose_from_cluster(localizer, qname, qcam, cluster_ids, features, matches)

                #
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster_id = id
                    best_inliers    = ret['num_inliers']
                #
                logs_clusters.append(log)
            
            # --> Best Cluster  TODO: What if it is not succesful 
            if best_cluster_id is not None:
                ret           = logs_clusters[best_cluster_id]['PnP_ret']
                poses[qname]  = (ret['qvec'], ret['tvec'])
            
            #
            logs['loc'][qname] = {
                'db': db_ids,
                'best_cluster': best_cluster_id,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:
            ret, log = pose_from_cluster(localizer, qname, qcam, db_ids, features, matches)
            
            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
            else:
                logger.warn("NOT Succesful")
                closest = sfm_model.images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
                
            log['covisibility_clustering'] = covisibility_clustering
            logs['loc'][qname] = log
        
        # query pose 3D visualization 
        if viewer:
            visualize_localization_3d(sfm_model, log, viewer=viewer)
    
    logger.info(f'Localized {len(poses)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    
    # Save results and logs to pkl file    
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    
    logger.info(f'Writing logs to {logs_path}...')
    
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    
    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_model', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)