# logger
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from loc.utils.colmap.database import COLMAPDatabase
from loc.utils.colmap.read_write_model import read_model
from loc.utils.geometry import compute_epipolar_errors
from loc.utils.io import OutputCapture, find_pair, path2key, read_pairs_dict
from loc.utils.readers import KeypointsLoader, MatchesLoader

from .base import Mapper

from loguru import logger

class ColmapMapper(Mapper):
    """ Colmap mapping class (SFM) 

    Args:
        workspace  ([str, Pathlib]) : path to sfm model
        cfg  (dict) : configuration parameters
    """

    def __init__(self, workspace: Path,  cfg: dict = {}):

        # cfg
        self.cfg = cfg

        #
        logger.info("init Colmap Mapper")

        # workspace
        self.workspace = workspace

        # images
        self.images_path = workspace / self.cfg.db.images
        
        

        #
        self.colmap_path = workspace / 'mapper'
        self.visloc_path = workspace / 'visloc'

        #
        self.database_path = self.visloc_path / 'database.db'

        #
        self.sfm_pairs_path = workspace / \
            str('sfm_pairs_' + str(self.cfg.mapper.num_covis) + '.txt')

        # read model
        # self.read_model(self.colmap_path)

    def load_model(self):
        """load colmap sift model

        Returns:
            pycolmap.Reconstruction: colmap sift reconstruction 
        """
        return pycolmap.Reconstruction(self.colmap_path)

    def load_visloc(self):
        """load visloc model

        Returns:
            pycolmap.Reconstruction: visloc reconstruction 
        """
        return pycolmap.Reconstruction(self.visloc_path)

    def read_model(self, model_path):
        """read colmap model

        Args:
            model_path ([str]): path to colmap reconstruction 
        """

        logger.info('reading Colmap model')

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        # read
        cameras, images, points3D = read_model(model_path)

        logger.info('model successfully loaded')

        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    def get_images(self):
        """get list of images

        Returns:
            List: list of images
        """
        return self.images

    def get_cameras(self):
        """get list of cameras

        Returns:
            List: list of cameras
        """
        return self.cameras

    def get_points3D(self):
        """get map 3D points

        Returns:
            List: list of map 3D points
        """
        return self.points3D

    def names_to_ids(self):
        """images names to ids mapping

        Returns:
            dict: names to ids mapping
        """
        model = self.load_model()
        return {image.name: i for i, image in model.images.items()}

    def covisible_pairs(self, sfm_pairs_path=None, num_covis=None):
        """get co-visible image pairs

        Args:
            num_covis (int, optional): number of covisible pairs . Defaults to None.

        Returns:
            str: path tp list of pairs (*.txt)
        """

        # read model
        self.read_model(self.colmap_path)
        
        if sfm_pairs_path is None:
            sfm_pairs_path = self.sfm_pairs_path

        if num_covis is None:
            num_covis = self.cfg.mapper.num_covis

        logger.info(f'searching for {num_covis} covisibility pairs ')

        #
        images = self.get_images()
        points3D = self.get_points3D()

        sfm_pairs = []
        for image_id, image in tqdm(images.items()):

            matched = image.point3D_ids != -1
            points3D_covis = image.point3D_ids[matched]

            # Histogram / Voting
            covis = defaultdict(int)
            for point_id in points3D_covis:
                for image_covis_id in points3D[point_id].image_ids:
                    if image_covis_id != image_id:
                        covis[image_covis_id] += 1

            if len(covis) == 0:
                logger.warning(
                    f'image {image_id} does not have any covisibility.')
                continue

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])

            # Sort and select
            if len(covis_ids) <= num_covis:
                top_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                # get covisible image ids with top k number of common matches
                ind_top = np.argpartition(covis_num, -num_covis)
                ind_top = ind_top[-num_covis:]  # unsorted top k
                ind_top = ind_top[np.argsort(-covis_num[ind_top])]

                top_covis_ids = [covis_ids[i] for i in ind_top]

                assert covis_num[ind_top[0]] == np.max(covis_num)

            # Collect pairs
            for i in top_covis_ids:
                pair = (image.name, images[i].name)
                sfm_pairs.append(pair)

        # save
        with open(sfm_pairs_path, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in sfm_pairs))

        logger.info(
            f'found {len(sfm_pairs)} covisible pairs saved {sfm_pairs_path}')

        return sfm_pairs_path

    def create_database(self):
        """create reconstruction database
        """

        #

        if self.database_path.exists():
            logger.info('The database already exists, deleting it.')
            self.database_path.unlink()

        # load model
        model = self.load_model()

        # create database
        db = COLMAPDatabase.connect(self.database_path)
        db.create_tables()

        # add cameras
        for i, camera in model.cameras.items():
            db.add_camera(camera.model_id, camera.width, camera.height,
                          camera.params, camera_id=i, prior_focal_length=True)

        # add images
        for i, image in model.images.items():
            db.add_image(image.name, image.camera_id, image_id=i)

        db.commit()
        db.close()

    def import_features(self, features_path):
        """import features to database

        Args:
            features_path (str): path to features data (*.h5) 
        """

        logger.info('importing features into the database...')

        self.keypoints_loader = KeypointsLoader(features_path)
        db = COLMAPDatabase.connect(self.database_path)

        image_ids = self.names_to_ids()

        for image_name, image_id in tqdm(image_ids.items()):
            keypoints, _ = self.keypoints_loader.load_as_numpy(image_name)
            keypoints += 0.5  # COLMAP origin
            db.add_keypoints(image_id, keypoints)

        db.commit()
        db.close()

    def import_matches(self,
                       pairs_path: Path,
                       matches_path: Path
                       ):
        """import matches to database

        Args:
            pairs_path (str): path to images pairs (*.txt)
            matches_path (str): path to features data (*.h5) 
        """

        logger.info('importing matches into the database...')

        #
        assert matches_path.exists, matches_path

        # loaders
        self.matches_loader = MatchesLoader(matches_path)

        with open(str(pairs_path), 'r') as f:
            pairs = [p.split() for p in f.readlines()]

        db = COLMAPDatabase.connect(self.database_path)

        image_ids = self.names_to_ids()

        matched = set()
        for name0, name1 in tqdm(pairs):

            id0, id1 = image_ids[name0], image_ids[name1]

            if len({(id0, id1), (id1, id0)} & matched) > 0:
                continue

            matches, scores = self.matches_loader.load_as_numpy(name0, name1)

            if self.cfg.pop("min_match_score", None):
                matches = matches[scores > self.cfg.pop("min_match_score", 0.)]

            db.add_matches(id0, id1, matches)
            matched |= {(id0, id1), (id1, id0)}

            if self.cfg.pop("skip_geometric_verification", None):
                db.add_two_view_geometry(id0, id1, matches)

        db.commit()
        db.close()

    def estimation_and_geometric_verification(self, pairs_path, verbose=False):
        """ colmap geometric verification for pair matches 
            pycolmap.verify_matches

        Args:
            pairs_path (str): path to images pairs (*.txt)
            verbose (bool, optional): console print. Defaults to False.
        """

        logger.info(
            'performing estimation and geometric verification of the matches...')

        # TODO: not pretty

        max_num_trials = self.cfg.mapper.max_num_trials
        min_inlier_ratio = self.cfg.mapper.min_inlier_ratio

        with OutputCapture(verbose):
            with pycolmap.ostream():
                pycolmap.verify_matches(self.database_path,
                                        pairs_path,
                                        max_num_trials=max_num_trials,
                                        min_inlier_ratio=min_inlier_ratio)

    def geometric_verification(self,
                               features_path: Path,
                               pairs_path: Path,
                               matches_path: Path
                               ) -> None:
        """geometric verification for pair matches 

        Args:
            features_path (str): path to features data (*.h5) 
            pairs_path (str): path to images pairs (*.txt)
            matches_path (str): path to features data (*.h5) 
        """
        assert features_path.exists,    features_path
        assert pairs_path.exists,       pairs_path
        assert matches_path.exists,     matches_path

        # loader
        self.keypoints_loader = KeypointsLoader(features_path)
        self.matches_loader = MatchesLoader(matches_path)

        # maximum epipolar error
        max_epip_error = self.cfg.mapper.max_epip_error

        logger.info('performing geometric verification of the matches...')

        #
        image_ids = self.names_to_ids()
        reference = self.load_model()

        #
        pairs = read_pairs_dict(pairs_path)
        db = COLMAPDatabase.connect(self.database_path)

        #
        inlier_ratios = []
        matched = set()
        for name0 in tqdm(pairs):

            id0 = image_ids[name0]

            image0 = reference.images[id0]
            cam0 = reference.cameras[image0.camera_id]

            kps0, noise0 = self.keypoints_loader.load_as_numpy(name0)

            if len(kps0) > 0:
                kps0 = np.stack(cam0.image_to_world(kps0))
            else:
                kps0 = np.zeros((0, 2))

            noise0 = 1.0 if noise0 is None else noise0

            for name1 in pairs[name0]:
                id1 = image_ids[name1]
                image1 = reference.images[id1]
                cam1 = reference.cameras[image1.camera_id]

                kps1, noise1 = self.keypoints_loader.load_as_numpy(name1)

                noise1 = 1.0 if noise1 is None else noise1

                if len(kps1) > 0:
                    kps1 = np.stack(cam1.image_to_world(kps1))
                else:
                    kps1 = np.zeros((0, 2))

                matches, _ = self.matches_loader.load_as_numpy(name0, name1)

                if len({(id0, id1), (id1, id0)} & matched) > 0:
                    continue

                matched |= {(id0, id1), (id1, id0)}

                if matches.shape[0] == 0:
                    db.add_two_view_geometry(id0, id1, matches)
                    continue

                qvec_01, tvec_01 = pycolmap.relative_pose(
                    image0.qvec, image0.tvec, image1.qvec, image1.tvec)
                _, errors0, errors1 = compute_epipolar_errors(
                    qvec_01, tvec_01, kps0[matches[:, 0]], kps1[matches[:, 1]])

                valid_matches = np.logical_and(
                    errors0 <= max_epip_error * noise0 / cam0.mean_focal_length(),
                    errors1 <= max_epip_error * noise1 / cam1.mean_focal_length())

                # TODO: We could also add E to the database, but we need
                # to reverse the transformations if id0 > id1 in utils/database.py.
                db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
                inlier_ratios.append(np.mean(valid_matches))

        logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                    np.mean(inlier_ratios) * 100,
                    np.median(inlier_ratios) * 100,
                    np.min(inlier_ratios) * 100,
                    np.max(inlier_ratios) * 100)

        db.commit()
        db.close()

    def triangulate_points(self, images_path, options={}, verbose=False):
        """triangulation 

        Args:
            options (dict, optional): triangulation options . Defaults to None.
            verbose (bool, optional): print console. Defaults to False.

        Returns:
            pycolmap.Reconstruction: reconstruction 
        """

        #
        output_path = self.visloc_path
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info('running 3D triangulation...')
        if options is None:
            options = {}

        # 
        model = self.load_model()

        #
        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstruction = pycolmap.triangulate_points(model,
                                                             self.database_path,
                                                             images_path,
                                                             output_path,
                                                             options=options)

        logger.info('finished the triangulation with statistics:\n%s',
                    reconstruction.summary())

        return reconstruction

    def run_sfm(self):

        #
        logger.info("run colmap incremental mapping")

        #
        database_path = self.colmap_path / "database.db"

        # if database_path.exists():
        #     database_path.unlink()

        # extraction options
        sift_options = {
            "num_workers": 16,
            "max_image_size": 640,
            "max_num_features": 2048
        }

        # extract features
        logger.info("colmap extract features")

        # pycolmap.extract_features(database_path,
        #                           self.images_path,
        #                           sift_options=sift_options,
        #                           verbose=True)

        # matcher options
        sift_matching_options = {
            "num_workers": 4,
        }
        exhaustive_options = {"block_size": 500}

        # match exhaustive
        logger.info("colmap match exhaustive")
        pycolmap.match_exhaustive(database_path,
                                  sift_options=sift_matching_options,
                                  matching_options=exhaustive_options)

        # mapper options
        mapper_options = {
            "num_workers": 16
        }

        # incremental mapping
        logger.info("colmap incremental mapping")
        maps = pycolmap.incremental_mapping(database_path,
                                            self.images_path,
                                            self.colmap_path,
                                            options=mapper_options)

        # write model
        logger.info(f"write colmap model {self.colmap_path}")
        maps[0].write(self.colmap_path)

        #
        return self.colmap_path
