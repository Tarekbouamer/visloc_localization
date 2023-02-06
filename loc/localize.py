import argparse
# logger
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pycolmap
from tqdm import tqdm

from loc.solvers.pnp import (AbsolutePoseEstimationPoseLib,
                             AbsolutePoseEstimationPyColmap)
from loc.utils.io import (dump_logs, read_pairs_dict,
                          write_poses_txt)

from loc.utils.readers import MatchesLoader, KeypointsLoader

logger = logging.getLogger("loc")


def covisibility_clustering(frame_ids: List[int], visloc_model: pycolmap.Reconstruction):
    """_summary_

    Args:
        frame_ids (List[int]): _description_
        visloc_model (pycolmap.Reconstruction): _description_

    Returns:
        _type_: _description_
    """

    clusters = []
    visited = set()

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
            observed = visloc_model.images[exploration_frame].points2D

            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D() for obs in visloc_model.points3D[p2D.point3D_id].track.elements
            }

            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    #
    clusters = sorted(clusters, key=len, reverse=True)

    return clusters


class DatasetLocalizer(object):
    """ general localizer

    Args:
        visloc_model (str): path to visloc model
        features (str): path to queries features
        matches (str): path to queries matches 
    """

    def __init__(self, visloc_model, features_path, matches_path, cfg={}):

        #
        self.cfg = cfg

        #
        logger.info('loading visloc model')
        if not isinstance(visloc_model, pycolmap.Reconstruction):
            visloc_model = pycolmap.Reconstruction(visloc_model)

        # TODO from config or args we select the pose estimator, add function make solver
        pose_estimator = AbsolutePoseEstimationPoseLib(visloc_model, cfg)

        self.visloc_model = visloc_model
        self.pose_estimator = pose_estimator

        assert features_path.exists(), features_path
        assert matches_path.exists(),  matches_path

        self.keypoints_loader = KeypointsLoader(features_path)
        self.matches_loader   = MatchesLoader(matches_path)

        self.covis_clustering = self.cfg.localize.covis_clustering

    def db_name_to_id(self):
        """name to ids

        Returns:
            dict: name to ids
        """
        return {img.name: i for i, img in self.visloc_model.images.items()}

    def pose_from_cluster(self, qname, qcam, db_ids, **kwargs):
        """cluster and find the best camera pose 

        Args:
            qname (str): query name
            qcam (numpy): query camera params
            db_ids (int): database ids

        Returns:
            dict: ret
            dict: log
        """

        # get 2D points
        kpq, _ = self.keypoints_loader.load_keypoints(qname)
        kpq += 0.5  # COLMAP coordinates

        # get 3D points
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0

        # 2D-3D macthing
        for i, db_id in enumerate(db_ids):

            image = self.visloc_model.images[db_id]

            #
            if image.num_points3D() == 0:
                logger.warning(f'zero 3D points for {image.name}.')
                continue

            # visible
            points3D_ids = np.array(
                [p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

            # matches
            matches, _ = self.matches_loader.load_matches(qname, image.name)
            matches = matches[points3D_ids[matches[:, 1]] != -1]

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
        ret = self.pose_estimator.estimate(
            kpq, mkp_idxs, mp3d_ids, qcam, **kwargs)

        ret['camera'] = {'model': qcam.model_name, 'width': qcam.width,
                         'height': qcam.height, 'params': qcam.params}

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                           for i in idxs for j in kp_idx_to_3D[i]]

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

    def __call__(self, queries, pairs_path, save_path=None):
        """localize

        Args:
            queries (str): queries cameras
            pairs_path (str): path to pairs
            save_path (str, optional): save directory. Defaults to None.
        """

        assert pairs_path.exists(),   pairs_path

        # load retrieval pairs
        logger.info('load retrievals pairs')
        retrieval_pairs = read_pairs_dict(pairs_path)

        #
        db_name_to_id = self.db_name_to_id()

        #
        poses = {}
        logs = {
            'retrieval': retrieval_pairs,
            'loc': {}
        }

        logger.info('starting localization')

        # -->
        for qname, qcam in tqdm(queries.items(), total=len(queries)):

            if qname not in retrieval_pairs:
                logger.warning(
                    f'no images retrieved for {qname} image. skipping...')
                continue

            db_names = retrieval_pairs[qname]
            db_ids = []

            for n in db_names:

                #
                if n not in db_name_to_id:
                    logger.debug(f'image {n} not in database')
                    continue
                db_ids.append(db_name_to_id[n])

            # empty
            if len(db_ids) < 1:
                logger.error(f"empty retrieval for {qname}")
                exit(0)

            # covisibility clustering
            if self.covis_clustering:
                clusters = covisibility_clustering(db_ids, self.visloc_model)

                best_inliers = 0
                best_cluster_id = None
                logs_clusters = []

                # pose from each cluster
                for id, cluster_ids in enumerate(clusters):

                    #
                    ret, log = self.pose_from_cluster(qname, qcam, cluster_ids)

                    #
                    if ret['success'] and ret['num_inliers'] > best_inliers:
                        best_cluster_id = id
                        best_inliers = ret['num_inliers']
                    #
                    logs_clusters.append(log)

                # --> Best Cluster
                if best_cluster_id is not None:
                    ret = logs_clusters[best_cluster_id]['PnP_ret']
                    poses[qname] = (ret['qvec'], ret['tvec'])

                #
                logs['loc'][qname] = {
                    'db': db_ids,
                    'best_cluster': best_cluster_id,
                    'log_clusters': logs_clusters,
                    'covis_clustering': self.covis_clustering
                }
            else:
                ret, log = self.pose_from_cluster(qname, qcam, db_ids)

                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                else:
                    logger.warn("not Succesful")
                    closest = self.visloc_model.images[db_ids[0]]
                    poses[qname] = (closest.qvec, closest.tvec)

                log['covis_clustering'] = self.covis_clustering
                logs['loc'][qname] = log

        if save_path is not None:
            write_poses_txt(poses, save_path)
            dump_logs(logs, save_path)

        logger.info(f'localized {len(poses)} / {len(queries)} images.')
        logger.info('done!')



class ImageLocalizer(object):
    """ general localizer

    Args:
        visloc_model (str): path to visloc model
        features (str): path to queries features
        matches (str): path to queries matches 
    """

    def __init__(self, visloc_model, retrieval, matcher, cfg={}):

        #
        self.cfg = cfg

        # 3D model
        logger.info('loading visloc model')
        if not isinstance(visloc_model, pycolmap.Reconstruction):
            visloc_model = pycolmap.Reconstruction(visloc_model)
        
        # pose estimator 
        pose_estimator = AbsolutePoseEstimationPoseLib(visloc_model, cfg)
        
        # retrieval
        self.retrieval = retrieval
        
        # matcher
        self.matcher = matcher

        self.covis_clustering = self.cfg.localize.covis_clustering
        
        self.visloc_model = visloc_model
        self.pose_estimator = pose_estimator

    def db_name_to_id(self):
        """name to ids

        Returns:
            dict: name to ids
        """
        return {img.name: i for i, img in self.visloc_model.images.items()}

    def pose_from_cluster(self, qname, qcam, db_ids, **kwargs):
        """cluster and find the best camera pose 

        Args:
            qname (str): query name
            qcam (numpy): query camera params
            db_ids (int): database ids

        Returns:
            dict: ret
            dict: log
        """

        # get 2D points
        kpq, _ = self.keypoints_loader.load_keypoints(qname)
        kpq += 0.5  # COLMAP coordinates

        # get 3D points
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0

        # 2D-3D macthing
        for i, db_id in enumerate(db_ids):

            image = self.visloc_model.images[db_id]

            #
            if image.num_points3D() == 0:
                logger.warning(f'zero 3D points for {image.name}.')
                continue

            # visible
            points3D_ids = np.array(
                [p.point3D_id if p.has_point3D() else -1 for p in image.points2D])

            # matches
            matches, _ = self.matches_loader.load_matches(qname, image.name)
            matches = matches[points3D_ids[matches[:, 1]] != -1]

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
        ret = self.pose_estimator.estimate(
            kpq, mkp_idxs, mp3d_ids, qcam, **kwargs)

        ret['camera'] = {'model': qcam.model_name, 'width': qcam.width,
                         'height': qcam.height, 'params': qcam.params}

        # mostly for logging and post-processing
        mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                           for i in idxs for j in kp_idx_to_3D[i]]

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

    def __call__(self, queries, pairs_path, save_path=None):
        """localize

        Args:
            queries (str): queries cameras
            pairs_path (str): path to pairs
            save_path (str, optional): save directory. Defaults to None.
        """

        assert pairs_path.exists(),   pairs_path

        # load retrieval pairs
        logger.info('load retrievals pairs')
        retrieval_pairs = read_pairs_dict(pairs_path)

        #
        db_name_to_id = self.db_name_to_id()

        #
        poses = {}
        logs = {
            'retrieval': retrieval_pairs,
            'loc': {}
        }

        logger.info('starting localization')

        # -->
        for qname, qcam in tqdm(queries.items(), total=len(queries)):

            if qname not in retrieval_pairs:
                logger.warning(
                    f'no images retrieved for {qname} image. skipping...')
                continue

            db_names = retrieval_pairs[qname]
            db_ids = []

            for n in db_names:

                #
                if n not in db_name_to_id:
                    logger.debug(f'image {n} not in database')
                    continue
                db_ids.append(db_name_to_id[n])

            # empty
            if len(db_ids) < 1:
                logger.error(f"empty retrieval for {qname}")
                exit(0)

            # covisibility clustering
            if self.covis_clustering:
                clusters = covisibility_clustering(db_ids, self.visloc_model)

                best_inliers = 0
                best_cluster_id = None
                logs_clusters = []

                # pose from each cluster
                for id, cluster_ids in enumerate(clusters):

                    #
                    ret, log = self.pose_from_cluster(qname, qcam, cluster_ids)

                    #
                    if ret['success'] and ret['num_inliers'] > best_inliers:
                        best_cluster_id = id
                        best_inliers = ret['num_inliers']
                    #
                    logs_clusters.append(log)

                # --> Best Cluster
                if best_cluster_id is not None:
                    ret = logs_clusters[best_cluster_id]['PnP_ret']
                    poses[qname] = (ret['qvec'], ret['tvec'])

                #
                logs['loc'][qname] = {
                    'db': db_ids,
                    'best_cluster': best_cluster_id,
                    'log_clusters': logs_clusters,
                    'covis_clustering': self.covis_clustering
                }
            else:
                ret, log = self.pose_from_cluster(qname, qcam, db_ids)

                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                else:
                    logger.warn("not Succesful")
                    closest = self.visloc_model.images[db_ids[0]]
                    poses[qname] = (closest.qvec, closest.tvec)

                log['covis_clustering'] = self.covis_clustering
                logs['loc'][qname] = log

        if save_path is not None:
            write_poses_txt(poses, save_path)
            dump_logs(logs, save_path)

        logger.info(f'localized {len(poses)} / {len(queries)} images.')
        logger.info('done!')



def do_localization(queries, pairs_path,
                    visloc_model, features_path, matches_path,
                    save_path=None,
                    cfg={}):
    """general localization 

    Args:
        # TODO: fix the queries side, not only queries cameras 
        queries (_type_): _description_

        pairs_path (str): path to pairs 
        visloc_model (str, pycolmap): path to visloc model
        features (str): path to queries local features 
        matches (str): path to queries local matches  
        save_path (str, optional): save_path directory. Defaults to None.
        cfg (dict, optional): configurations. Defaults to {}.
    """

    loc = Localizer(visloc_model=visloc_model,
                    features_path=features_path,
                    matches_path=matches_path,
                    cfg=cfg)

    # run
    loc(queries, pairs_path, save_path=save_path)
