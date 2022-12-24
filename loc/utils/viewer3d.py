import argparse
import numpy as np
import open3d
import pycolmap

from copy import deepcopy
import os 

from loc.utils.colmap.read_write_model import read_model, write_model, qvec2rotmat, rotmat2qvec


def draw_camera(K, R, t, w, h, scale=1, color=[1.0, 0.0, 0.0]):
    """Create axis, plane and pyramed geometries in Open3D format.
    
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K.copy() / scale
    Kinv = np.linalg.inv(K)

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))

    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)

    # points in pixel
    points_pixel = [
        [0, 0, 0],
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1],
    ]

    # pixel to camera coordinate system
    points = [Kinv @ p for p in points_pixel]

    # image plane
    width  = abs(points[1][0]) + abs(points[3][0])
    height = abs(points[1][1]) + abs(points[3][1])
    plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6)
    plane.paint_uniform_color(color)
    plane.translate([points[1][0], points[1][1], scale])
    plane.transform(T)

    # pyramid
    points_in_world = [(R @ p + t) for p in points]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
    ]
    colors = [color for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points_in_world),
        lines=open3d.utility.Vector2iVector(lines))
    line_set.colors = open3d.utility.Vector3dVector(colors)

    # return as list in Open3D format
    return [axis, plane, line_set]


class Visualizer:
    def __init__(self):
        self.cameras = []
        self.images = []
        self.points3D = []
        self.vis = None

    def read_model(self, path, ext=""):
        self.cameras, self.images, self.points3D = read_model(path, ext)

    def add_points(self, min_track_len=3, remove_statistical_outlier=True):
        pcd = open3d.geometry.PointCloud()
        

        xyz = []
        rgb = []
        for point3D in self.points3D.values():
            track_len = len(point3D.point2D_idxs)
            if track_len < min_track_len:
                continue
            
            xyz.append(point3D.xyz)
            rgb.append(point3D.rgb / 255.)

        pcd.points = open3d.utility.Vector3dVector(xyz)
        pcd.colors = open3d.utility.Vector3dVector(rgb)

        # remove obvious outliers
        if remove_statistical_outlier:
            [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

        # open3d.visualization.draw_geometries([pcd])
        self.vis.add_geometry(pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def add_cameras(self, scale=1):
        frames = []
        for img in self.images.values():
            # rotation
            R = qvec2rotmat(img.qvec)

            # translation
            t = img.tvec

            # invert
            t = -R.T @ t
            R = R.T

            # intrinsics
            cam = self.cameras[img.camera_id]

            if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            else:
                raise Exception("Camera model not supported")

            # intrinsics
            K = np.identity(3)
            K[0, 0] = fx
            K[1, 1] = fy
            K[0, 2] = cx
            K[1, 2] = cy

            # create axis, plane and pyramed geometries that will be drawn
            cam_model = draw_camera(K, R, t, cam.width, cam.height, scale)
            frames.extend(cam_model)

        # add geometries to visualizer
        for i in frames:
            self.vis.add_geometry(i)

    def create_window(self):
        self.vis = open3d.visualization.Visualizer()
        self.vis.create_window()

    def show(self,render_option_path=None ):
        self.vis.poll_events()
        # self.vis.update_renderer()
        self.vis.clear_geometries()
        self.vis.get_render_option().load_from_json(render_option_path)
        self.add_points(min_track_len=3)
    
        self.add_cameras(scale=0.4)
        self.vis.run()
        self.vis.destroy_window()


class VisualizerGui:
    def __init__(self):
        self.vis = None
    
    def read_model(self, path):
        self.model = pycolmap.Reconstruction(path)

    def create_window(self):
        self.gui = open3d.visualization.gui.Application.instance
        self.gui.initialize()
        self.vis = open3d.visualization.O3DVisualizer(title="SFM", width=2048, height=1024)
    
    def pcd_from_colmap(rec, min_track_length=3, max_reprojection_error=100):
        points = []
        colors = []
        for p3D in rec.points3D.values():
            if p3D.track.length() < min_track_length:
                continue

            points.append(p3D.xyz)
            colors.append(p3D.color/255.)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(np.stack(points))
        pcd.colors = open3d.utility.Vector3dVector(np.stack(colors))
        return pcd
    
    def add_points(self):
        
        pcd = self.pcd_from_colmap(self.model)
        self.vis.add_geometry('pcd', pcd)
        
    def add_cameras(self):
        camera_lines = {}
        
        for camera in self.model.cameras.values():
            camera_lines[camera.camera_id] = open3d.geometry.LineSet.create_camera_visualization(
                camera.width, 
                camera.height, 
                camera.calibration_matrix(), 
                np.eye(4), 
                scale=0.5)
        
        for image in self.model.images.values():
            T = np.eye(4)
            T[:3, :4] = image.inverse_projection_matrix()
            cam = deepcopy(camera_lines[image.camera_id]).transform(T)
            cam.paint_uniform_color([1.0, 0.0, 0.0])  # red
            
            self.vis.add_geometry(image.name, cam)
    
    def show(self):
        
        self.add_cameras()
        self.add_points()
  
        self.vis.reset_camera_to_default() 
        self.vis.scene_shader = self.vis.UNLIT
        self.vis.point_size = 1
        # self.vis.enable_raw_mode(False)

        self.gui.add_window(self.vis)
        self.gui.run()
  




