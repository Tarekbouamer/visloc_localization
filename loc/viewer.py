

import multiprocessing as mp 
from multiprocessing import Process, Queue, Value
import random

from  collections import Set
import numpy as np

import pypangolin as pango
import OpenGL.GL as gl

# logger
import logging
logger = logging.getLogger("loc")


# [4x4] homogeneous inverse T^-1 from [4x4] T     
def inv_T(T):
    
    ret     = np.eye(4)
    R_T     = T[:3,:3].T
    t       = T[:3,3]
    
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t
    
    return ret 


kUiWidth                = 180
kDefaultPointSize       = 5
kViewportWidth          = 1024
kViewportHeight         = 768
kViewportHeight         = 550
kDrawCameraPrediction   = False   
kDrawReferenceCamera    = True 
  
kMinWeightForDrawingCovisibilityEdge = 100

MIN_TRACK_LENGTH = 4

class Viewer3DMapElement(object): 
    def __init__(self):
        # query
        self.query_pose = []

        
        # database
        self.db_poses = []
        self.points3D = []
        self.colors3D = []       
        
        self.covisibility_graph = []
        self.spanning_tree = []        
        self.loops = []    
    
              
class Viewer3D(object):
    
    def __init__(self):
        self.map       = Queue()
        self.cameras   = Queue() 
        
        self.cam_state = None 
        self.map_state = None 
              
        self._is_running    = Value('i', 1)
        self._is_paused     = Value('i', 1)
        
        self.vp = Process(target=self.viewer_thread, 
                          args=(self.map, 
                                self.cameras, 
                                self._is_running, 
                                self._is_paused,))
        
        self.vp.daemon = True
        self.vp.start()

    def quit(self):
        self._is_running.value = 0
        self.vp.join()
        print('3D viewer stopped')   
        
    def is_paused(self):
        return (self._is_paused.value == 1)       

    def viewer_thread(self, map, cameras, is_running, is_paused):
        
        self.viewer_init(kViewportWidth, kViewportHeight)
        
        while not pango.ShouldQuit() and (is_running.value == 1):
            self.viewer_refresh(map, cameras, is_paused)
        
        print('Quitting viewer...')    

    def viewer_init(self, w, h):
        
        # Create windows
        win = pango.CreateWindowAndBind("pySimpleDisplay", w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
    
        
        # Camera initial position 
        viewpoint_x = 0
        viewpoint_y = -40
        viewpoint_z = -80
        viewpoint_f = 1000
         
        self.proj       = pango.ProjectionMatrix(w, h, viewpoint_f, viewpoint_f, w//2, h//2, 0.1, 5000)
        self.look_view  = pango.ModelViewLookAt(viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0)
        self.scam       = pango.OpenGlRenderState(self.proj, self.look_view)
        
        # Handler
        self.handler    = pango.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pango.CreateDisplay()
        self.dcam.SetBounds(pango.Attach(0), pango.Attach(1), pango.Attach.Pix(kUiWidth), pango.Attach(1), -640.0 / 480.0,)
        self.dcam.SetHandler(self.handler)
        
        
        # Create Panel
        self.panel = pango.CreatePanel('ui')
        self.panel.SetBounds(pango.Attach(0), pango.Attach(1), pango.Attach(0), pango.Attach.Pix(kUiWidth))     
        
        self.ui = pango.Var("ui")
        self.ui.Button          = False
        self.ui.Follow          = False
        self.ui.Cams            = True
        self.ui.Covisibility    = True
        self.ui.SpanningTree    = True
        self.ui.Grid            = True
        self.ui.Pause           = False

        # self.ui.slider      =   (3, pango.VarMeta(0, 5))
        self.ui.log_slider  =   (3.0,   pango.VarMeta(1, 1e4,   logscale=True)  )
        self.ui.slider      =   (2,     pango.VarMeta(1, 10)                    ) 
        
        # Focus        
        self.Twc    = pango.OpenGlMatrix()
        self.Twc.SetIdentity()

        # Follow cameras
        self.do_follow      = True
        self.is_following   = True 
    
    def viewer_refresh(self, map, cameras, is_paused):
        
        print("viewer_refresh")

        while not cameras.empty():
            self.cam_state = cameras.get()
        
        while not map.empty():
            self.map_state = map.get()
        
        if self.ui.Pause:
            is_paused.value = 0  
        else:
            is_paused.value = 1 
        
        # Check follow mechanism 
        if self.ui.Follow and self.is_following:
            self.scam.Follow(self.Twc, True)
        elif self.ui.Follow and not self.is_following:
            self.scam.SetModelViewMatrix(self.look_view)
            self.scam.Follow(self.Twc, True)
            self.is_following = True
        elif not self.ui.Follow and self.is_following:
            self.is_following = False   
        

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        
        self.dcam.Activate(self.scam)
        
        # grid is ON/OFF    
        if self.ui.Grid:
            Viewer3D.drawPlane()   
            
     
        if self.map_state is not None:
            # Draw query pose
            # if self.map_state.query_pose is not None:
            #     # in blue
            #     gl.glColor3f(0.0, 0.0, 1.0)
            #     gl.glLineWidth(2)                
            #     self.drawCamera(self.map_state.query_pose)
            #     gl.glLineWidth(1)                
            #     self.updateTwc(self.map_state.query_pose)
            
            # key frame pose       
            if len(self.map_state.db_poses) >= 1:
                # draw keyframe poses in green
                if self.ui.Cams:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    self.drawCameras(self.map_state.db_poses[:])
                    
            # points
            if len(self.map_state.points3D)>0:
                gl.glPointSize(self.ui.slider)
                
                if  len(self.map_state.colors3D) > 1:
                    gl.glColor3f(1.0, 0.0, 0.0)
                    
                self.drawPointsColor(self.map_state.points3D, 
                                     self.map_state.colors3D)  
                 

        # Update Frame
        pango.FinishFrame()
    
    def drawPoints(self, points) :
        
        gl.glBegin(gl.GL_POINTS);
        
        for p in points:
            gl.glVertex3d(p[0], p[1], p[2]);
    
        gl.glEnd();
    
    def drawPointsColor(self,  points, colors):
        
        gl.glBegin(gl.GL_POINTS);
        
        for p, c in zip(points, colors):
            gl.glColor3f(   c[0], c[1], c[2]);
            gl.glVertex3d(  p[0], p[1], p[2]);

        gl.glEnd();
        
    def drawCameras(self, cameras,  w=1, h_ratio=0.75, z_ratio=0.5):
        for cam in cameras:
            self.drawCamera(cam, w=w, h_ratio=h_ratio, z_ratio=z_ratio) 
              
    def drawCamera(self, camera,  w=1, h_ratio=0.75, z_ratio=0.5):
        
        h = w * h_ratio
        z = w * z_ratio

        gl.glPushMatrix()
        gl.glMultTransposeMatrixd(camera);

        gl.glBegin(gl.GL_LINES)
        
        gl.glVertex3f(0,0,0)
        gl.glVertex3f(w,h,z)
        
        gl.glVertex3f(0,0,0)
        gl.glVertex3f(w,-h,z)
        
        gl.glVertex3f(0,0,0)
        gl.glVertex3f(-w,-h,z)
        
        gl.glVertex3f(0,0,0)
        gl.glVertex3f(-w,h,z)

        gl.glVertex3f(w,h,z)
        gl.glVertex3f(w,-h,z)

        gl.glVertex3f(-w,h,z)
        gl.glVertex3f(-w,-h,z)

        gl.glVertex3f(-w,h,z)
        gl.glVertex3f(w,h,z)

        gl.glVertex3f(-w,-h,z)
        gl.glVertex3f(w,-h,z)
        
        gl.glEnd()

        gl.glPopMatrix()
    
    def draw_localization(self, sfm_model=None, log=None, seed=2):
        if sfm_model is None:
            return 
        
        # Map State
        map_state  = Viewer3DMapElement()

        # Get database ids 
        db_ids       = log["db"]
        points3D_ids = log["points3D_ids"]
        PnP_ret      = log["PnP_ret"]
        
        if len(db_ids)==0  or len(points3D_ids)==0:
            return

        # Add retrieved images 
        for i, db_id in enumerate(db_ids):
            image = sfm_model.images[db_id]
            
            # Pose
            rot_mat     =   getRotmat(image.qvec)
            rotmatINV   =   np.transpose(rot_mat)
                
            trans_vec   =   image.tvec
            trans_vec   =   -rotmatINV.dot(trans_vec)
            
            pose   = np.eye(4)
            pose[:3, :3]   = rotmatINV
            pose[:3, 3]    = trans_vec
                    
            map_state.db_poses.append(pose) 

        # Add retrieved point3D 
        for j, point3d_id in enumerate(points3D_ids):
                
            pt3D = sfm_model.points3D[j]
                
            if pt3D.track.length() < MIN_TRACK_LENGTH:
                continue

            map_state.points3D.append(pt3D.xyz)
            map_state.colors3D.append(pt3D.color/255.)  
        
        # Add Query
        print(PnP_ret.keys())
        if PnP_ret["success"]:
            rot_mat     =   getRotmat(PnP_ret["qvec"])
            rotmatINV   =   np.transpose(rot_mat)
                
            trans_vec   =   PnP_ret["tvec"]
            trans_vec   =   -rotmatINV.dot(trans_vec)
            
            pose   = np.eye(4)
            pose[:3, :3]   = rotmatINV
            pose[:3, 3]    = trans_vec
                    
            map_state.query_pose = pose 
            
        # numpy
        
        map_state.db_poses      = np.array(map_state.db_poses)
        map_state.points3D      = np.array(map_state.points3D)
        map_state.colors3D      = np.array(map_state.colors3D) 
        
        self.map.put(map_state)

    def draw_sfm(self, sfm_model=None, seed=2):
        
        logger.info("draw sfm ")
        import pycolmap
        
        sfm_model = pycolmap.Reconstruction(sfm_model)  
        
        map_state  = Viewer3DMapElement()
        
        # -->
        for i in sfm_model.images:
        
            #
            image = sfm_model.images[i]

            # pose
            rot_mat   = getRotmat(image.qvec)
            rotmatINV = np.transpose(rot_mat)
                
            trans_vec = image.tvec
            trans_vec = -rotmatINV.dot(trans_vec)
            
            pose        = np.eye(4)
            pose[:3,:3] = rotmatINV
            pose[:3,3]  = trans_vec
                    
            map_state.db_poses.append(pose) 
            
        # -->
        for j in sfm_model.points3D:
            
            pt3D = sfm_model.points3D[j]

            if pt3D.track.length() < MIN_TRACK_LENGTH:
                continue

            map_state.points3D.append(pt3D.xyz)
            map_state.colors3D.append(pt3D.color/255.)
        
        
        # numpy
        map_state.db_poses    = np.array(map_state.db_poses)
        map_state.points3D    = np.array(map_state.points3D)
        map_state.colors3D    = np.array(map_state.colors3D) 
     
        # print(map_state.db_poses)
        # print(map_state.points3D)
        # print(map_state.colors3D)
        self.map.put(map_state)              

    def updateTwc(self, pose):
        self.Twc = pango.OpenGlMatrix(pose)
        
    @staticmethod
    def drawPlane(num_divs=200, div_size=10):
        

        minx = -num_divs*div_size
        minz = -num_divs*div_size
        maxx = num_divs*div_size
        maxz = num_divs*div_size

        gl.glColor3f(0.7,0.7,0.7)
        gl.glBegin(gl.GL_LINES)
        
        for n in range(2*num_divs):
            gl.glVertex3f(minx+div_size*n,0,minz)
            gl.glVertex3f(minx+div_size*n,0,maxz)
            gl.glVertex3f(minx,0,minz+div_size*n)
            gl.glVertex3f(maxx,0,minz+div_size*n)
        
        gl.glEnd()


def getRotmat(qRot):
    rot = np.zeros((3, 3))
    
    rot[0,0]    =   1.0-2.0*(qRot[2]*qRot[2]+qRot[3]*qRot[3])
    rot[0,1]    =   2.0*(qRot[1]*qRot[2]-qRot[3]*qRot[0])
    rot[0,2]    =   2.0*(qRot[1]*qRot[3]+qRot[2]*qRot[0])
    
    rot[1,0]    =   2.0*(qRot[1]*qRot[2]+qRot[3]*qRot[0])
    rot[1,1]    =   1.0-2.0*(qRot[1]*qRot[1]+qRot[3]*qRot[3])
    rot[1,2]    =   2.0*(qRot[2]*qRot[3]-qRot[1]*qRot[0])

    rot[2,0]    =   2.0*(qRot[1]*qRot[3]-qRot[2]*qRot[0])
    rot[2,1]    =   2.0*(qRot[2]*qRot[3]+qRot[1]*qRot[0])
    rot[2,2]    =   1.0-2.0*(qRot[1]*qRot[1]+qRot[2]*qRot[2])
    
    return rot