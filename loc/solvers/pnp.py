import pycolmap
import poselib
import torch

# logger
import logging
logger = logging.getLogger("loc")


class AbsolutePoseEstimation:
    def __init__(self, sfm_model, config=None):
        self.sfm_model = sfm_model

        config = {"estimation": {"ransac": {"max_error": 18}}}

        self.config = {**self.default_config, **config}

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return x

    def estimate(self):
        raise NotImplementedError

    def __repr__(self):
        cls = self.__class__.__name__

        msg = f'{cls}('
        for k, v in self.config.items():
            msg += f'\t {k}: {v}'
        msg += f')'

        return msg


class AbsolutePoseEstimationPyColmap(AbsolutePoseEstimation):

    default_config = {
        "estimation": {"ransac": {"max_error": 12}}}

    def __init__(self, sfm_model, config=None):
        super().__init__(sfm_model=sfm_model, config=config)

    def estimate(self, points2D_all, points2D_idxs, points3D_id, query_camera):

        # 
        points2D_all = self._to_numpy(points2D_all)
        points2D_idxs = self._to_numpy(points2D_idxs)

        # 
        points2D = points2D_all[points2D_idxs]
        points3D = [self.sfm_model.points3D[j].xyz for j in points3D_id]

        # estimation
        ret = pycolmap.absolute_pose_estimation(points2D,
                                                points3D,
                                                query_camera,
                                                estimation_options=self.config.get(
                                                    'estimation', {}),
                                                refinement_options=self.config.get('refinement', {}))
        return ret


class AbsolutePoseEstimationPoseLib(AbsolutePoseEstimation):

    default_config = {
        'ransac': {'max_reproj_error': 12.0, 'max_epipolar_error': 1.0},
        'bundle': {'max_iterations': 100}
    }

    def __init__(self, sfm_model, config=None):
        super().__init__(sfm_model=sfm_model, config=config)

    def estimate(self, points2D_all, points2D_idxs, points3D_id, camera):

        # 
        points2D_all = self._to_numpy(points2D_all)
        points2D_idxs = self._to_numpy(points2D_idxs)
        
        points2D = points2D_all[points2D_idxs]
        points3D = [self.sfm_model.points3D[j].xyz for j in points3D_id]

        # camera model
        camera = {'model':  camera.model_name,  'width':  camera.width,
                  'height': camera.height,      'params': camera.params}
        
        # print(points2D)
        # print(points3D)
        
        # print(points2D.shape)
        # print(points3D.shape)

        # print(camera)
        # input()
        # estimation
        pose, info = poselib.estimate_absolute_pose(points2D, points3D, camera,
                                                    self.config.get(
                                                        'ransac', {}),
                                                    self.config.get('bundle', {}))

        #
        ret = {**info}
        ret['qvec'] = pose.q
        ret['tvec'] = pose.t
        ret['success'] = True

        return ret
