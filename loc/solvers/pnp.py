import pycolmap
import poselib
import torch

# logger
import logging
logger = logging.getLogger("loc")


class AbsolutePoseEstimation:
    def __init__(self, sfm_model, cfg=None):
        self.sfm_model = sfm_model

        cfg = {"estimation": {"ransac": {"max_error": 18}}}

        self.cfg = {**self.default_cfg, **cfg}

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        else:
            return x

    def estimate(self):
        raise NotImplementedError

    def __repr__(self):
        msg  = f" {self.__class__.__name__}"
        msg += f" ransac: max_reproj_error {self.cfg.solver.ransac.max_reproj_error}"
        msg += f" max_epipolar_error {self.cfg.solver.ransac.max_epipolar_error}"
        msg += f" bundle: max_iterations {self.cfg.solver.bundle.max_iterations}"
        return msg


class AbsolutePoseEstimationPyColmap(AbsolutePoseEstimation):

    default_cfg = {
        "estimation": {"ransac": {"max_error": 12}}}

    def __init__(self, sfm_model, cfg=None):
        super().__init__(sfm_model=sfm_model, cfg=cfg)

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
                                                estimation_options=self.cfg.get(
                                                    'estimation', {}),
                                                refinement_options=self.cfg.get('refinement', {}))
        return ret


class AbsolutePoseEstimationPoseLib(AbsolutePoseEstimation):

    default_cfg = {
        'ransac': {'max_reproj_error': 12.0, 'max_epipolar_error': 1.0},
        'bundle': {'max_iterations': 100}
    }

    def __init__(self, sfm_model, cfg=None):
        super().__init__(sfm_model=sfm_model, cfg=cfg)

    def estimate(self, points2D_all, points2D_idxs, points3D_id, camera):

        # 
        points2D_all = self._to_numpy(points2D_all)
        points2D_idxs = self._to_numpy(points2D_idxs)
        
        points2D = points2D_all[points2D_idxs]
        points3D = [self.sfm_model.points3D[j].xyz for j in points3D_id]

        # camera model
        camera = {'model':  camera.model_name,  'width':  camera.width,
                  'height': camera.height,      'params': camera.params}
        
        # estimation
        pose, info = poselib.estimate_absolute_pose(points2D, points3D, camera,
                                                    self.cfg.get(
                                                        'ransac', {}),
                                                    self.cfg.get('bundle', {}))

        #
        ret = {**info}
        ret['qvec'] = pose.q
        ret['tvec'] = pose.t
        ret['success'] = True

        return ret
