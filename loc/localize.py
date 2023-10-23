
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pycolmap
from loguru import logger
from tqdm import tqdm

from loc.matchers import Matcher
from loc.solvers.pnp import AbsolutePoseEstimationPoseLib
from loc.utils.io import dump_logs, read_pairs_dict, write_poses_txt
from loc.utils.readers import KeypointsReader, LocalFeaturesReader, MatchesReader


def covisibility_clustering(frame_ids: List[int], 
                            visloc_model: pycolmap.Reconstruction):
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
                for p2D in observed if p2D.has_point3D() \
                    for obs in visloc_model.points3D[p2D.point3D_id].track.elements
            }

            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    #
    clusters = sorted(clusters, key=len, reverse=True)

    return clusters


class Localizer:
    def __init__(self,
                 visloc_model,
                 cfg) -> None:

        #
        self.cfg = cfg

        #
        logger.info('loading visloc model')
        if not isinstance(visloc_model, pycolmap.Reconstruction):
            visloc_model = pycolmap.Reconstruction(visloc_model)

        self.visloc_model = visloc_model

        # pose estimator
        self.pose_estimator = AbsolutePoseEstimationPoseLib(visloc_model, cfg)

        #
        self.name_to_id = {img.name: i for i,
                           img in self.visloc_model.images.items()}

    def get_cluster_ids(self, cluster_names):
        
        # db indices
        db_ids = []

        for name in cluster_names:

            if name not in self.name_to_id:
                logger.debug(f'image {name} not found in database')
                continue
            #
            db_ids.append(self.name_to_id[name])

        return db_ids

    def pose_from_cluster(self):
        raise NotImplementedError


def wrap_keys_with_extenstion(data: Dict,
                              ext="0"
                              ):
    return {k+ext: v for k, v in data.items()}


class DatasetLocalizer(Localizer):

    def __init__(self, visloc_model, features_path, matches_path, cfg={}) -> None:
        super().__init__(visloc_model, cfg)

        assert features_path.exists(), features_path
        assert matches_path.exists(),  matches_path

        self.keypoints_loader = KeypointsReader(features_path)
        self.matches_loader = MatchesReader(matches_path)

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
        kpq, _ = self.keypoints_loader.load_as_numpy(qname)
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
            matches, _ = self.matches_loader.load_as_numpy(qname, image.name)
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
            db_ids = self.get_cluster_ids(db_names)

            # empty
            if len(db_ids) < 1:
                logger.error(f"empty retrieval for {qname}")
                exit(0)

            # covisibility clustering
            if self.cfg.localize.covis_clustering:
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
                    'covis_clustering': self.cfg.localize.covis_clustering
                }
            else:
                ret, log = self.pose_from_cluster(qname, qcam, db_ids)

                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                else:
                    logger.warn("not Succesful")
                    closest = self.visloc_model.images[db_ids[0]]
                    poses[qname] = (closest.qvec, closest.tvec)

                log['covis_clustering'] = self.cfg.localize.covis_clustering
                logs['loc'][qname] = log

        if save_path is not None:
            write_poses_txt(poses, save_path)
            dump_logs(logs, save_path)

        logger.info(f'localized {len(poses)} / {len(queries)} images.')
        logger.info('done!')


class ImageLocalizer(Localizer):

    def __init__(self, visloc_model, extractor, retrieval, db_features_path, 
                 features_path, 
                 matches_path, cfg={}
                 ) -> None:
        super().__init__(visloc_model, cfg)

        # extractor
        self.extractor = extractor

        # retrieval
        self.retrieval = retrieval

        # matcher
        self.matcher = Matcher(cfg)

        # 
        self.db_features_loader = LocalFeaturesReader(save_path=db_features_path)

        self.matches_loader = MatchesReader(matches_path)
        self.keypoints_loader = KeypointsReader(features_path)

        #
        self.poses = {}
        self.logs = {
            'loc': {}
        }

    def pose_from_cluster(self, qname, qcam, q_preds, db_ids):

        # get 2D points COLMAP coordinates
        kpq = q_preds["keypoints"]
        kpq += 0.5 

        # get 3D points
        kp_idx_to_3D = defaultdict(list)
        kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
        num_matches = 0

        # perform 2D-3D macthing
        for i, db_id in enumerate(db_ids):

            # get db image
            image = self.visloc_model.images[db_id]

            # skip empty views
            if image.num_points3D() == 0:
                logger.warning(f'zero 3D points for {image.name}.')
                continue

            # get visible points
            points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1 \
                for p in image.points2D])

            # compute query db matches
            matches = self.compute_matches(q_preds, image.name)
            matches = matches[points3D_ids[matches[:, 1]] != -1]

            for idx, m in matches:
                id_3D = points3D_ids[m]
                kp_idx_to_3D_to_db[idx][id_3D].append(i)
                # avoid duplicate observations
                if id_3D not in kp_idx_to_3D[idx]:
                    kp_idx_to_3D[idx].append(id_3D)
            
            #        
            num_matches += len(matches)


        #
        idxs = list(kp_idx_to_3D.keys())
        mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
        mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]

        # pose estimation
        ret = self.pose_estimator.estimate(kpq, mkp_idxs, mp3d_ids, qcam)

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

    def compute_matches(self, q_preds, db_name):
        
        # query
        q_preds = wrap_keys_with_extenstion(q_preds, ext="0")

        # db
        db_preds = self.db_features_loader.load(db_name)
        db_preds = wrap_keys_with_extenstion(db_preds, ext="1")
        
        # match 
        preds = self.matcher.match_pair({**q_preds, **db_preds})    
        matches = preds["matches"].cpu().numpy()

        #
        _idx = np.where(matches != -1)[0]
        matches = np.stack([_idx, matches[_idx]], -1)
        
        return matches

    def localize_image(self, data: Dict):

        #
        qname, qcam = data['name'][0], data['camera']

        # find best image pairs
        pairs_names = self.retrieval(data)
        
        # extract locals
        q_preds = self.extractor.extract_image(
            data, normalize=False,  gray=True)
        
        # db indices and names
        db_names = [x[1] for x in pairs_names]
        db_ids = self.get_cluster_ids(db_names)

        # empty
        if len(db_ids) < 1:
            logger.error(f"empty retrieval for {data['name']}")
            exit(0)

        # pose
        ret, log = self.pose_from_cluster(
            qname, qcam, q_preds, db_ids)

        if ret['success']:
            qpose = (ret['qvec'], ret['tvec'])
        else:
            closest = self.visloc_model.images[db_ids[0]]
            qpose = (closest.qvec, closest.tvec)

        #
        self.poses[qname] = qpose

        log['covis_clustering'] = self.cfg.localize.covis_clustering
        self.logs['loc'][qname] = log

        return qpose


def do_localization(queries, pairs_path,
                    visloc_model, features_path, matches_path,
                    save_path=None,
                    cfg={}):

    loc = Localizer(visloc_model=visloc_model,
                    features_path=features_path,
                    matches_path=matches_path,
                    cfg=cfg)

    # run
    loc(queries, pairs_path, save_path=save_path)
