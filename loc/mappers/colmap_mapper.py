import os
import logging
from pathlib import Path
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from loc.utils.read_write_model import read_model
from loc.colmap.database import COLMAPDatabase

# from xrloc.map.read_write_model import read_images_binary
# from xrloc.maimport p.read_write_model import read_points3d_binary
# from xrloc.map.read_write_model 


# logger
import logging
logger = logging.getLogger("loc")


class ColmapMapper(object):

    def __init__(self, model_path):
        
        # model exsists
        if isinstance(model_path, str):
            model_path = Path(model_path)
        
        #
        self.model_path = model_path
        
        # read model
        self.read_model(model_path)
                
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
    
    def covisible_pairs(self, num_matches=5):

        logger.info('extracting image pairs from covisibility')
        
        sfm_pairs = []
        for image_id, image in tqdm(self.images.items()):
            matched        = image.point3D_ids != -1
            points3D_covis = image.point3D_ids[matched]

            # Histogram / Voting
            covis = defaultdict(int)
            for point_id in points3D_covis:
                for image_covis_id in self.points3D[point_id].image_ids:
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

                pair = (image.name, self.images[i].name)
                sfm_pairs.append(pair)
        
        logger.info(f'found {len(sfm_pairs)} image pairs.')
        self.sfm_pairs = sfm_pairs
        
        return sfm_pairs
    
    
    def create_database(self, database_path):
      
        if database_path.exists():
            logger.info('The database already exists, deleting it.')
            database_path.unlink()

        db = COLMAPDatabase.connect(database_path)
        db.create_tables()

        for i, camera in self.cameras.items():
            print(camera)
            db.add_camera(camera.model_id, camera.width, camera.height, camera.params, camera_id=i, prior_focal_length=True)

        for i, image in self.images.items():
            db.add_image(image.name, image.camera_id, image_id=i)

        db.commit()
        db.close()
        return {image.name: i for i, image in self.images.items()}
      
      
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

    def visible_points(self, image_ids):
        """Get visible 3D point ids for given image id list.
        Args:
            image_ids (list[int]): The image id list
        Returns:
            np.array(int64): The 3D point ids that are visible for
                input images
        """
        set_point3d_ids = set()
        for id in image_ids:
            point3d_ids = self.images[id].point3D_ids
            valid_point3d_ids = point3d_ids[point3d_ids != -1]
            set_point3d_ids.update(valid_point3d_ids)
        mp_point3d_ids = np.array(list(set_point3d_ids))
        return mp_point3d_ids

    def point3d_at(self, point3d_id):
        """Get 3D point coordinate.
        Args:
            point3d_id (int): The 3D point id
        Returns:
            np.array(float): 3D point coordinate
        """
        return self.point3ds[point3d_id]

    def image_at(self, image_id):
        """Get image.
        Args:
            image_id (int): The image id
        Returns:
            Image: The image with id == image_id
        """
        return self.images[image_id]

    def point3d_coordinates(self, point3d_ids):
        """Get the coordinates of multi 3D points.
        Args:
            point3d_ids (array[int]): The point 3D ids
        Returns:
            np.array(float, 3*N): The coordinates
        """
        coordinates = np.array([
            self.point3ds[point3d_id].xyz for point3d_id in point3d_ids
        ]).transpose()
        return coordinates

    def point3d_features(self, point3d_ids):
        """Get the descriptors of multi 3D points.
        Args:
            point3d_ids (array[int]): The point 3D ids
        Returns:
            np.array(float, dim*N): The descriptors
        """
        features = np.array([
            self.features[point3d_id] for point3d_id in point3d_ids
        ]).transpose()
        return features