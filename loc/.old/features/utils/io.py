# logger
from loguru import logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from loc.utils.io import load_aachen_intrinsics

from loguru import logger

_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']


def read_image(path: Path):
    # read
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f'error in read image {path}.')

    # BGR to RGB
    if len(image.shape) == 3:
        image = image[:, :, ::-1]

    #
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]

    return image, size


def show_cv_image(image):
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()


def cv_to_tensor(image):
  
    image = image.transpose((2, 0, 1))
    
    # normalize [0, 1.]
    image = image / 255.

    # out
    image = image.astype(np.float32)

    image = torch.from_numpy(image).unsqueeze(0)

    return image
  
  
def show_cv_image_keypoints(image, kpts, mode="bgr"):
  
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()      
    
    if len(image.shape) == 4:
        image = image[0] 
        
    # CHW --> HWC
    image = image.transpose((1, 2, 0))
    
    plt.imshow(image)
   
    plt.scatter(kpts[:, 0], kpts[:, 1], c="r", s=3, linewidths=0)
    
    plt.show()
