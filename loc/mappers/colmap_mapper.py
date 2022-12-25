import logging
from pathlib import Path
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from loc.colmap.database import COLMAPDatabase
from loc.utils.colmap.read_write_model import read_model
from loc.utils.io import parse_name, find_pair, get_keypoints, get_matches, parse_retrieval, OutputCapture
from loc.utils.geometry import compute_epipolar_errors

from .base import Mapper

import pycolmap

# logger
import logging
logger = logging.getLogger("loc")


class ColmapMapper(Mapper):
    """ Colmap mapping class (SFM) 

    Args:
        workspace  ([str, Pathlib]) : path to sfm model
        cfg  (dict) : configuration parameters
    """ 
    
    def __init__(self, data_path, workspace, cfg={}):
        
        # cfg
        self.cfg = cfg
        
        #
        logger.info("init Colmap Mapper")
        
        #
        self.workspace      = workspace
        self.images_path    = data_path / self.cfg.db.images
        
        self.colmap_model_path     = workspace / 'colmap_model'
        self.visloc_model_path     = workspace / 'visloc_model'  

        self.colmap_model_path.mkdir(parents=True, exist_ok=True)
        self.visloc_model_path.mkdir(parents=True, exist_ok=True)
        #
        self.database_path   = self.visloc_model_path / 'database.db'
        self.sfm_pairs_path = workspace / str('sfm_pairs_' + str(self.cfg.mapper.num_covis) + '.txt') 
                         
        # read model
        self.read_model(self.colmap_model_path)
        
    def load_model(self):
        """load colmap sift model

        Returns:
            pycolmap.Reconstruction: colmap sift reconstruction 
        """        
        return pycolmap.Reconstruction(self.colmap_model_path)
 
    def load_visloc(self):
        """load visloc model

        Returns:
            pycolmap.Reconstruction: visloc reconstruction 
        """        
        return pycolmap.Reconstruction(self.visloc_model_path)  
     
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

        self.cameras  = cameras
        self.images   = images
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
    
    def covisible_pairs(self, num_covis=None):
        """get co-visible image pairs

        Args:
            num_covis (int, optional): number of covisible pairs . Defaults to None.

        Returns:
            str: path tp list of pairs (*.txt)
        """        

        if num_covis is None:
            num_covis = self.cfg.mapper.num_covis
        
        logger.info(f'searching for {num_covis} covisibility pairs ')
        
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
            if len(covis_ids) <= num_covis:
                top_covis_ids = covis_ids[np.argsort(-covis_num)]
            else:
                # get covisible image ids with top k number of common matches
                ind_top     = np.argpartition(covis_num, -num_covis)
                ind_top     = ind_top[-num_covis:]  # unsorted top k
                ind_top     = ind_top[np.argsort(-covis_num[ind_top])]
                
                top_covis_ids = [covis_ids[i] for i in ind_top]
                
                assert covis_num[ind_top[0]] == np.max(covis_num)

            # Collect pairs 
            for i in top_covis_ids:

                pair = (image.name, images[i].name)
                sfm_pairs.append(pair)
        
        # save
        with open(self.sfm_pairs_path, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in sfm_pairs))

        logger.info(f'found {len(sfm_pairs)} covisible pairs saved {self.sfm_pairs_path}')
          
        return self.sfm_pairs_path
    
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
            db.add_camera(camera.model_id, camera.width, camera.height, camera.params, camera_id=i, prior_focal_length=True)

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
        
        db = COLMAPDatabase.connect(self.database_path)
        
        image_ids = self.names_to_ids()
        
        for image_name, image_id in tqdm(image_ids.items()):        
            keypoints = get_keypoints(features_path, image_name)
            keypoints += 0.5  # COLMAP origin
            db.add_keypoints(image_id, keypoints)

        db.commit()
        db.close()  

    def import_matches(self, pairs_path, matches_path):
        """import matches to database

        Args:
            pairs_path (str): path to images pairs (*.txt)
            matches_path (str): path to features data (*.h5) 
        """        
        
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
        
        logger.info('performing estimation and geometric verification of the matches...')
        
        # TODO: not pretty
        
        max_num_trials      =self.cfg.mapper.max_num_trials
        min_inlier_ratio    = self.cfg.mapper.min_inlier_ratio

        with OutputCapture(verbose):
            with pycolmap.ostream():
                pycolmap.verify_matches(self.database_path, 
                                        pairs_path, 
                                        max_num_trials=max_num_trials, 
                                        min_inlier_ratio=min_inlier_ratio)

    def geometric_verification(self, features_path, pairs_path, matches_path):
        """geometric verification for pair matches 

        Args:
            features_path (str): path to features data (*.h5) 
            pairs_path (str): path to images pairs (*.txt)
            matches_path (str): path to features data (*.h5) 
        """
        
        # maximum epipolar error
        max_epip_error = self.cfg.mapper.max_epip_error 
        
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
                    errors0 <= max_epip_error * noise0 / cam0.mean_focal_length(),
                    errors1 <= max_epip_error * noise1 / cam1.mean_focal_length())

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

    def triangulate_points(self, options={}, verbose=False):
        """triangulation 

        Args:
            options (dict, optional): triangulation options . Defaults to None.
            verbose (bool, optional): print console. Defaults to False.

        Returns:
            pycolmap.Reconstruction: reconstruction 
        """        
        
        # 
        output_path = self.visloc_model_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        #
        image_path = self.images_path
        
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
    
    def run(self):
        raise NotImplementedError