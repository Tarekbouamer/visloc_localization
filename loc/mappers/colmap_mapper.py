import os
import logging
from pathlib import Path
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from loc.colmap.database import COLMAPDatabase
from loc.utils.read_write_model import read_model
from loc.utils.io import parse_name, find_pair, get_keypoints, get_matches, parse_retrieval, OutputCapture
from loc.utils.geometry import compute_epipolar_errors


import pycolmap

# logger
import logging
logger = logging.getLogger("loc")


class ColmapMapper(object):
    
    default_cfg = {
        
    }

    def __init__(self, model_path):
        
        #
        logger.info("init ColmapMapper")
        
        # model exsists
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        #
        self.model_path = model_path
        
        # read model
        self.read_model(model_path)
        
    def load_model(self):
        return pycolmap.Reconstruction(self.model_path)
    
    def read_model(self, model_path):
        
        logger.info('reading Colmap model')
        
        if not model_path.exists():
           raise FileNotFoundError(model_path)

        # read
        cameras, images, points3D = read_model(model_path)
        logger.info('model successfully loaded')

        self.cameras  = cameras
        self.images   = images
        self.points3D = points3D
    
    def get_images(self):
        return self.images
    
    def get_cameras(self):
        return self.cameras
    
    def get_points3D(self):
        return self.points3D
    
    def set_database_path(self, database_path):
        self.database_path = database_path
    
    def names_to_ids(self):  
        model = self.load_model()
        return {image.name: i for i, image in model.images.items()}  
    
    def covisible_pairs(self, num_matches=5):

        logger.info('extracting image pairs from covisibility')
        
        #
        images      = self.get_images()
        points3D    = self.get_points3D()
        
        sfm_pairs = []
        for image_id, image in tqdm(images.items()):

            matched        = image.point3D_ids != -1
            points3D_covis = image.point3D_ids[matched]

            # Histogram / Voting
            covis = defaultdict(int)
            for point_id in points3D_covis:
                for image_covis_id in points3D[point_id].image_ids:
                    if image_covis_id != image_id:
                        covis[image_covis_id] += 1

            if len(covis) == 0:
                logger.warning (f'image {image_id} does not have any covisibility.')
                continue

            covis_ids = np.array(list(covis.keys()))
            covis_num = np.array([covis[i] for i in covis_ids])
            
            # Sort and select        
            if len(covis_ids) <= num_matches:
                top_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                # get covisible image ids with top k number of common matches
                ind_top     = np.argpartition(covis_num, -num_matches)
                ind_top     = ind_top[-num_matches:]  # unsorted top k
                ind_top     = ind_top[np.argsort(-covis_num[ind_top])]
                
                top_covis_ids = [covis_ids[i] for i in ind_top]
                
                assert covis_num[ind_top[0]] == np.max(covis_num)

            # Collect pairs 
            for i in top_covis_ids:

                pair = (image.name, images[i].name)
                sfm_pairs.append(pair)
        
        logger.info(f'found {len(sfm_pairs)} image pairs.')
        
        # 
        self.sfm_pairs = sfm_pairs
        
        return sfm_pairs
    
    def create_database(self, database_path):
        
        # 
        self.set_database_path(database_path)
        
        if database_path.exists():
            logger.info('The database already exists, deleting it.')
            database_path.unlink()
        
        # load model
        model = self.load_model()

        # create database
        db = COLMAPDatabase.connect(database_path)
        db.create_tables()
        
        # add cameras
        for i, camera in model.cameras.items():
            db.add_camera(camera.model_id, camera.width, camera.height, camera.params, camera_id=i, prior_focal_length=True)

        # add images
        for i, image in model.images.items():
            db.add_image(image.name, image.camera_id, image_id=i)

        db.commit()
        db.close()
            
    def import_features(self, features_path):
        
        logger.info('importing features into the database...')
        
        db = COLMAPDatabase.connect(self.database_path)
        
        image_ids = self.names_to_ids()
        
        for image_name, image_id in tqdm(image_ids.items()):        
            keypoints = get_keypoints(features_path, image_name)
            keypoints += 0.5  # COLMAP origin
            db.add_keypoints(image_id, keypoints)

        db.commit()
        db.close()  

    def import_matches(self, pairs_path, matches_path, min_match_score=None, skip_geometric_verification=False):
        
        logger.info('importing matches into the database...')

        with open(str(pairs_path), 'r') as f:
            pairs = [p.split() for p in f.readlines()]

        db = COLMAPDatabase.connect(self.database_path)

        image_ids = self.names_to_ids()

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
             
    def covisible_images(self, image_id, num_covisble_point=1):
        """Get co-visible images.
        Args:
            image_id (int): Image id
            num_covisble_point (int): The number of co-visible 3D point
        Returns:
            list[int]: Co-visible image ids
        """
        covisible_images_to_num_points = {}
        if image_id not in self.images:
            logging.error('Image id {} not exist in reconstruction. The reason '
                          'may be you did not specify images_bin_path when '
                          'creating database.bin '.format(image_id))
            return []
        point3d_ids = self.images[image_id].point3D_ids
        for point3d_id in point3d_ids:
            if point3d_id == -1:
                continue
            image_ids = self.point3ds[point3d_id].image_ids
            for id in image_ids:
                if id in covisible_images_to_num_points:
                    covisible_images_to_num_points[id] += 1
                else:
                    covisible_images_to_num_points[id] = 1

        covisible_pairs = [(id, covisible_images_to_num_points[id])
                           for id in covisible_images_to_num_points]

        covisible_pairs = sorted(covisible_pairs,
                                 key=lambda k: k[1],
                                 reverse=True)

        image_ids = [
            id for id, num_point in covisible_pairs
            if num_point >= num_covisble_point and id != image_id
        ]

        return [image_id] + image_ids

    def estimation_and_geometric_verification(self, pairs_path, verbose=False):
        
        logger.info('performing estimation and geometric verification of the matches...')
        
        with OutputCapture(verbose):
            with pycolmap.ostream():
                pycolmap.verify_matches(self.database_path, pairs_path, max_num_trials=20000, min_inlier_ratio=0.1)

    def geometric_verification(self, features_path, pairs_path, matches_path, max_error=4.0):

        logger.info('performing geometric verification of the matches...')
        
        # 
        image_ids = self.names_to_ids()
        reference = self.load_model()
        
        #
        pairs = parse_retrieval(pairs_path)
        db = COLMAPDatabase.connect(self.database_path)

        #
        inlier_ratios = []
        matched = set()
        for name0 in tqdm(pairs):
            
            id0     = image_ids[name0]
            
            image0  = reference.images[id0]
            cam0    = reference.cameras[image0.camera_id]
            
            kps0, noise0    = get_keypoints(features_path, name0, return_uncertainty=True)
            
            if len(kps0) > 0:
                kps0 = np.stack(cam0.image_to_world(kps0))
            else:
                kps0 = np.zeros((0, 2))        
            
            noise0 = 1.0 if noise0 is None else noise0

            for name1 in pairs[name0]:
                id1 = image_ids[name1]
                image1 = reference.images[id1]
                cam1 = reference.cameras[image1.camera_id]
                
                kps1, noise1 = get_keypoints(features_path, name1, return_uncertainty=True)
                noise1 = 1.0 if noise1 is None else noise1

                if len(kps1) > 0:
                    kps1 = np.stack(cam1.image_to_world(kps1))
                else:
                    kps1 = np.zeros((0, 2))
                    
                matches = get_matches(matches_path, name0, name1)[0]

                if len({(id0, id1), (id1, id0)} & matched) > 0:
                    continue
                
                matched |= {(id0, id1), (id1, id0)}

                if matches.shape[0] == 0:
                    db.add_two_view_geometry(id0, id1, matches)
                    continue
                
                qvec_01, tvec_01    = pycolmap.relative_pose(
                    image0.qvec, image0.tvec, image1.qvec, image1.tvec)
                _, errors0, errors1 = compute_epipolar_errors(
                    qvec_01, tvec_01, kps0[matches[:, 0]], kps1[matches[:, 1]])
                

                valid_matches = np.logical_and(
                    errors0 <= max_error * noise0 / cam0.mean_focal_length(),
                    errors1 <= max_error * noise1 / cam1.mean_focal_length())

                # TODO: We could also add E to the database, but we need
                # to reverse the transformations if id0 > id1 in utils/database.py.
                db.add_two_view_geometry(id0, id1, matches[valid_matches, :])
                inlier_ratios.append(np.mean(valid_matches))
        
        logger.info('mean/med/min/max valid matches %.2f/%.2f/%.2f/%.2f%%.',
                    np.mean(inlier_ratios)      * 100, 
                    np.median(inlier_ratios)    * 100,
                    np.min(inlier_ratios)       * 100, 
                    np.max(inlier_ratios)       * 100)

        db.commit()
        db.close()       

    def run_triangulation(self, image_path, output_path, options=None, verbose=False):
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info('running 3D triangulation...')
        if options is None:
            options = {}    
        
        model = self.load_model()
        
        with OutputCapture(verbose):
            with pycolmap.ostream():
                reconstruction = pycolmap.triangulate_points(model, 
                                                             self.database_path, 
                                                             image_path, 
                                                             output_path,
                                                             options=options)

        logger.info('finished the triangulation with statistics:\n%s', reconstruction.summary())
        
        return reconstruction                 
    