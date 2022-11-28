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

from loc.utils.io import load_aachen_intrinsics

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
                transforms.Normalize(mean=mean, std=std)
                ])
    
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
    
    def __call__(self, img):
        
        # resize
        img = self.__resize__(img)

        #
        img = self.postprocessing(img)           

        return dict(img=img)


class ImagesFromList(data.Dataset):
    
    def __init__(self, images_path, cameras_path=None, split=None, transform=None, max_size=None): 
        
        self.max_size       = max_size
        self.split          = split        
        self.images_path    = images_path
        self.cameras_path   = cameras_path
        
        # Load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.images_path).glob('**/'+ ext)) 
                                        
        if len(paths) == 0:
            raise ValueError(f'Could not find any image in path: {self.images_path}.')
            
        self.images_fn = sorted(list(set(paths)))
                    
        logger.info('Found %s images in %s', len(self.images_fn), self.images_path ) 
        
        # Load intrinsics
        if self.cameras_path:
            self.cameras = load_aachen_intrinsics(self.cameras_path)
            logger.info('Imported %s from %s ', len(self.cameras), self.cameras_path)
            
        # transform
        self.transform = ImagesTransform(max_size=max_size) if transform is None else transform

    def __len__(self):
        return len(self.images_fn)

    def resize_image(self, img, size):
        img     = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        return  img     
    
    def get_names(self):
        return [ self.split + "/" + str(p.relative_to(self.images_path)) for p in self.images_fn] 
         
    def load_img(self, img_path):
          
        # for truncated images
        ImageFile.LOAD_TRUNCATED_IMAGES = True     
        
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
          
        return img
    
    def get_cameras(self,):
        if hasattr(self, "cameras"):
            return self.cameras     
        else:
            None          
    
    def get_name(self,  _path):
        return path.basename(_path)   
     
    def __getitem__(self, item):
        #
        out = {}
        
        #
        img_path    = self.images_fn[item]
        img_name    = self.get_name(img_path)
        
        # pil
        img  = self.load_img(img_path)
        
        # transform
        if self.transform is not None:
            out = self.transform(img)
                        
        # Dict
        out["name"]         = img_name
        
        return out