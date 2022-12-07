import collections.abc as collections
from os import path

import torch
import torch.utils.data as data

import numpy as np

from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageFile

from    torchvision.transforms import functional as tfn
import  torchvision.transforms as transforms
from  timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pycolmap 
import cv2
import h5py

from loc.utils.io import load_aachen_intrinsics, parse_name

# logger
import logging
logger = logging.getLogger("loc")


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)

    return image


def list_h5_names(path):
    names = []
    
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
      
    return list(set(names))


def get_keypoints(path: Path, name: str) -> np.ndarray:
    with h5py.File(str(path), 'r') as hfile:
        p = hfile[name]['keypoints'].__array__()
    return p
  
def parse_image_list(path, with_intrinsics=False, logger=None):
    images = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if len(line) == 0 or line[0] == '#':
                continue
            name, *data = line.split()
            if with_intrinsics:
                model, width, height, *params = data
                params = np.array(params, float)
                cam = pycolmap.Camera(model, int(width), int(height), params)
                images.append((name, cam))
            else:
                images.append(name)

    assert len(images) > 0
    
    if logger:
        logger.info(f'Imported {len(images)} images from {path.name}')
    return 
  
def parse_image_lists(paths, with_intrinsics=False):
    images = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0
    for lfile in files:
        images += parse_image_list(lfile, with_intrinsics=with_intrinsics)
    return images
  



class ImagesTransform:
  
    def __init__(self, max_size, 
                 mean=IMAGENET_DEFAULT_MEAN,
                 std=IMAGENET_DEFAULT_STD):
        
        # max_size
        self.max_size = max_size
        
        # preprocessing 
        self.postprocessing = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=mean, std=std)
                ])
        # cv
        self.resize_force = False
        self.interpolation = 'cv2_area'
        
    def __resize__(self, img):
        
        # scale
        width, height   = img.size
        max_size        = max(width, height)
        
        # scale
        scale = self.max_size / max_size
        
        #
        out_size = tuple(int(dim * scale) for dim in img.size)

        # resize
        return img.resize(out_size, resample=Image.BILINEAR)
    
    def resize(self, image, size, interp=cv2.INTER_AREA):
        if self.max_size and (self.resize_force or max(size) > self.max_size):
            scale       = self.resize_max / max(size)
            size_new    = tuple(int(round(x*scale)) for x in size)
            
            h, w = image.shape[:2]
            if interp == cv2.INTER_AREA and (w < size_new[0] or h < size_new[1]):
                interp = cv2.INTER_LINEAR
            image = cv2.resize(image, size_new, interpolation=interp)
   
    def __call__(self, img):
        
        # resize
        # img = self.__resize__(img)
        img = self.resize(img)

        #
        # img = self.postprocessing(img)           

        return dict(img=img)




class ImagesFromList(data.Dataset):
    
    def __init__(self, root, data_cfg, split=None, transform=None, max_size=None, **kwargs): 
        
        # cfg
        cfg =  data_cfg[split]
        
        # options
        self.max_size       = max_size
        self.root           = Path(root)
        
        # camera path
        self.cameras_path = None
        if cfg['cameras'] is not None:
            self.cameras_path   = Path(root) / str(cfg['cameras'])
        
        # images path
        if cfg['images'] is not None:
            self.images_path  = Path(root) / str(cfg['images']) 

        # split path
        self.split_images_path  = self.images_path / str(split)

        # load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.split_images_path).glob('**/'+ ext)) 
                                        
        if len(paths) == 0:
            raise ValueError(f'could not find any image in path: {self.split_images_path}.')
        
        # sort   
        self.images_fn = sorted(list(set(paths)))
        
        # all names 
        self.names = [i.relative_to(self.images_path).as_posix() for i in self.images_fn]         
        
        # load intrinsics
        if self.cameras_path is not None:
            self.cameras = load_aachen_intrinsics(self.cameras_path)
        else:
            self.cameras = None
                    
        # gray scale
        self.gray = kwargs.pop('gray', False)
            
        # transform numpy ->  tensor
        self.transform = ImagesTransform(max_size=max_size) if transform is None else transform
        
        
        logger.info(f'found {len(self.images_fn)} images with {self.num_cameras()} intrinsics') 

    def __len__(self):
        return len(self.images_fn)
    
    def read_image(path, grayscale=False):
        
        if grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        
        image = cv2.imread(str(path), mode)
        
        if image is None:
            raise ValueError(f'Cannot read image {path}.')
        
        if not grayscale and len(image.shape) == 3:
            image = image[:, :, ::-1]  # BGR to RGB
        # 
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        
        return image , size  
    
    def load_img(self, img_path):
          
        # truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True     
        
        # open 
        with open(img_path, 'rb') as f:
            if self.gray:
                img = Image.open(f).convert('L')
            else:
                img = Image.open(f).convert('RGB')
                
        return img
    
    def num_cameras(self):
        return len(self.cameras) if self.cameras is not None else None
    
    def get_cameras(self):
        return self.cameras     

    def get_names(self):
        return self.names 
             
    def get_name(self, _path):
        name = _path.relative_to(self.images_path).as_posix()
        return name
     
    def __getitem__(self, item):
        #
        out = {}
        
        #
        img_path    = self.images_fn[item]
        img_name    = self.get_name(img_path)
                        
        # pil
        img  = self.load_img(img_path)
        size = img.size
        
        # cv
        img, size = self.read_image(img_path, grayscale=self.gray)
        
        # transform
        if self.transform is not None:
            out = self.transform(img)
        else:
            out['img'] = img
                        
        # dict
        out["name"]  = img_name
        out["size"]  = np.array(size, dtype=float)
        
        return out