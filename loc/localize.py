import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm
import pycolmap

from typing import Dict, List, Union, Tuple
from collections import defaultdict

from loc.utils.io import parse_retrieval_file, get_keypoints, get_matches

from loc.solvers.pnp import AbsolutePoseEstimationPoseLib, AbsolutePoseEstimationPyColmap

# logger
import logging
logger = logging.getLogger("loc")


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
    


def pose_from_cluster(localizer, qname, query_camera, db_ids, features_path, matches_path, **kwargs):
    
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
            logger.warning(f'No 3D points found for {image.name}.')
            continue
        
        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

                
        matches, _  = get_matches(matches_path, qname, image.name)
        matches     = matches[points3D_ids[matches[:, 1]] != -1]
        
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
    ret = localizer.estimate(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)
        
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


class Localizer:
    default_cfg = {}
    def __init__(self, sfm_model, features, matches, cfg=None):
        
        #
        self.cfg = {**self.default_cfg, **cfg}
        
        #
        logger.info('reading the sfm model') 
        if not isinstance(sfm_model, pycolmap.Reconstruction):
            sfm_model = pycolmap.Reconstruction(sfm_model)
        
        #
        pose_estimator = AbsolutePoseEstimationPoseLib(sfm_model, cfg)

        self.sfm_model  = sfm_model
        self.pose_estimator = pose_estimator
        
        assert features.exists(), features
        assert matches.exists(),  matches       
        
        self.features   = features
        self.matches    = matches
        
        self.covisibility_clustering = False


    def db_name_to_id(self):
        return {img.name: i for i, img in self.sfm_model.images.items()}  
    
    def pose_from_cluster(self, qname, qcam, db_ids, **kwargs):
        
        # get 2D points
        kpq = get_keypoints(self.features, qname)
        kpq += 0.5  # COLMAP coordinates

        # get 3D points
        kp_idx_to_3D        = defaultdict(list)
        kp_idx_to_3D_to_db  = defaultdict(lambda: defaultdict(list))
        num_matches         = 0
        
        # 2D-3D macthing
        for i, db_id in enumerate(db_ids):
            
            image = self.sfm_model.images[db_id]
            
            #
            if image.num_points3D() == 0:
                logger.warning(f'zero 3D points for {image.name}.')
                continue
            
            # visible 
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

            # matches     
            matches, _  = get_matches(self.matches, qname, image.name)
            matches     = matches[points3D_ids[matches[:, 1]] != -1]
            
            num_matches += len(matches)

            for idx, m in matches:
                id_3D = points3D_ids[m]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
        
        #
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
        
        # pose estimation  
        ret = self.pose_estimator.estimate(kpq, mkp_idxs, mp3d_ids, qcam, **kwargs)
            
        ret['camera'] = {'model': qcam.model_name, 'width': qcam.width, 'height':qcam.height, 'params': qcam.params}

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j]) for i in idxs for j in kp_idx_to_3D[i]]
        
        #
        log = {
            'db': db_ids,
            'PnP_ret': ret,
            'keypoints_query': kpq[mkp_idxs],
            'points3D_ids': mp3d_ids,
            'points3D_xyz': None, 
            'num_matches': num_matches,
            'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
        }
        
        return ret, log
        
    def __call__(self, queries, retrieval_pairs_path, save_path=None):
        
        assert retrieval_pairs_path.exists(),   retrieval_pairs_path
        
        # load retrieval pairs
        logger.info('load retrievals pairs') 
        retrieval_pairs = parse_retrieval_file(retrieval_pairs_path)
        
        #
        db_name_to_id = self.db_name_to_id()
        
        # 
        poses = {}
        logs  = {'retrieval': retrieval_pairs,   'loc': {}}
        
        logger.info('starting localization')
        
        # -->
        for qname, qcam in tqdm(queries.items(), total=len(queries)):
            
            if qname not in retrieval_pairs:
                logger.warning(f'no images retrieved for {qname} image. skipping...')
                continue
            
            db_names = retrieval_pairs[qname]
            db_ids   = []
            
            for n in db_names:
                if n not in db_name_to_id:
                    logger.debug(f'image {n} not in database')
                    continue
                #    
                db_ids.append(db_name_to_id[n])
    
            #      
            if len(db_ids) < 1:
                logger.error(f"empty retrieval for {qname}")
                exit(0)  
                
            # covisibility clustering
            if self.covisibility_clustering:
                clusters = do_covisibility_clustering(db_ids, self.sfm_model)
            
                best_inliers    = 0
                best_cluster_id = None
                logs_clusters   = []
                
                # pose from each cluster
                for id, cluster_ids in enumerate(clusters):
                    
                    ret, log = self.pose_from_cluster(qname, qcam, cluster_ids)

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
                    'covisibility_clustering': self.covisibility_clustering
                }
            else:
                ret, log = self.pose_from_cluster(qname, qcam, cluster_ids)
                
                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                else:
                    logger.warn("not Succesful")
                    closest = self.sfm_model.images[db_ids[0]]
                    poses[qname] = (closest.qvec, closest.tvec)
                                
                log['covisibility_clustering'] = self.covisibility_clustering
                logs['loc'][qname] = log
        
        pass
        

def main(sfm_model, queries, retrieval_pairs_path, features,
         matches, results,
         ransac_thresh=12,
         covisibility_clustering=False,
         prepend_camera_name=False,
         config=None, 
         viewer=None):

    # 
    # assert retrieval_pairs_path.exists(),   retrieval_pairs_path
    # assert features.exists(),               features
    # assert matches.exists(),                matches
    
    # # load retrieval pairs
    # logger.info('load retrievals pairs') 
    # retrievals = parse_retrieval_file(retrieval_pairs_path)

    # logger.info('reading the 3D model...') 
    # if not isinstance(sfm_model, pycolmap.Reconstruction):
    #     sfm_model = pycolmap.Reconstruction(sfm_model)
    # db_name_to_id = {img.name: i for i, img in sfm_model.images.items()}
    
    # localizer
    localizer = AbsolutePoseEstimationPoseLib(sfm_model, config)
    
    # poses = {}
    # logs = {
    #     'features': features,
    #     'matches': matches,
    #     'retrieval': retrievals,
    #     'loc': {},
    # }
    
    
    logger.info('starting localization...')
    
    for qname, qcam in tqdm(queries.items(), total=len(queries)):

        #  
        if qname not in retrievals:
            logger.debug(f'no images retrieved for query image {qname}. skipping...')
            continue
        
        # geo-verification
        db_names = retrievals[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.debug(f'image {n} was retrieved but not in database')
                continue
            #    
            db_ids.append(db_name_to_id[n])
    
        #      
        if len(db_ids) < 1:
            logger.error("empty retrieval")
            exit(0)
        
        # covisibility clustering --> Hloc
        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, sfm_model)
            
            best_inliers    = 0
            best_cluster_id = None
            logs_clusters   = []
            
            # for each cluser compute the query pose
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
                logger.warn("not Succesful")
                closest = sfm_model.images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
                            
            log['covisibility_clustering'] = covisibility_clustering
            logs['loc'][qname] = log
    
    # 
    write_poses_txt(poses, results)
    dump_logs(logs, results)
    
    logger.info(f'localized {len(poses)} / {len(queries)} images.')
    logger.info('done!')


    
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