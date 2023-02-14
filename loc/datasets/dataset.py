# logger
import logging
from os import path
from pathlib import Path
from typing import Any, Dict, Tuple, List

import cv2
import numpy as np
import pycolmap
from torch.utils.data import Dataset

from loc.utils.io import load_aachen_intrinsics

logger = logging.getLogger("loc")


_EXT = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']


class ImagesFromList(Dataset):

    def __init__(self,
                 root: Path,
                 split: str = None,
                 cfg={}) -> None:

        #
        self.split = split
        self.root = Path(root)

        if split == "db":
            self.max_size = cfg.retrieval.max_size
        elif split == "query":
            self.max_size = cfg.extractor.max_size
        else:
            self.max_size = None

        # camera path
        self.cameras_path = None
        if cfg[split]['cameras'] is not None:
            self.cameras_path = Path(root) / str(cfg[split]['cameras'])

        # images split path
        self.split_images_path = Path(root) / str(cfg[split]["images"])
        self.images_rel_path = self.split_images_path.parents[0]

        # load images
        paths = []
        for ext in _EXT:
            paths += list(Path(self.split_images_path).glob('**/' + ext))

        if len(paths) == 0:
            raise ValueError(
                f'could not find any image in path: {self.split_images_path}.')

        # sort
        self.images_fn = sorted(list(set(paths)))

        # all names
        self.names = [i.relative_to(self.images_rel_path).as_posix()
                      for i in self.images_fn]

        # load intrinsics
        if self.cameras_path is not None:
            self.cameras = load_aachen_intrinsics(self.cameras_path)
        else:
            self.cameras = None

        # cv
        self.interpolation = 'cv2_area'

        #
        logger.info(
            f'found {len(self.images_fn)} images with {self.num_cameras()} intrinsics')

    def __len__(self) -> None:
        return len(self.images_fn)

    def resize(self,
               image: np.ndarray,
               size: Tuple[int],
               interp: Any = cv2.INTER_AREA
               ) -> np.ndarray:

        if self.max_size and max(size) > self.max_size:

            scale = self.max_size / max(size)
            size_new = tuple(int(round(x*scale)) for x in size)

            h, w = image.shape[:2]

            if interp == cv2.INTER_AREA and (w < size_new[0] or h < size_new[1]):
                interp = cv2.INTER_LINEAR

            #
            image = cv2.resize(image, size_new, interpolation=interp)

        return image

    def read_image(self, 
                   path: Path
                   ) -> Tuple[np.ndarray, np.ndarray]:

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

    def num_cameras(self) -> int:
        return len(self.cameras) if self.cameras is not None else None

    def get_cameras(self) -> Dict[str, pycolmap.Camera]:
        return self.cameras

    def get_names(self) -> List[str]:
        return self.names

    def get_name(self, 
                 _path: Path
                 ) -> str:
        name = _path.relative_to(self.images_rel_path).as_posix()
        return name

    def __getitem__(self, item) -> Dict:

        out = {}
        
        #
        img_path = self.images_fn[item]
        img_name = self.get_name(img_path)

        # read image
        img, size = self.read_image(img_path)

        # resize image
        img = self.resize(img, size=size)

        # --> CHW
        img = img.transpose((2, 0, 1))

        # normalize [0, 1.]
        img = img / 255.

        # out
        out["img"]  = img.astype(np.float32)
        out["name"] = img_name
        out["size"] = np.array(size, dtype=np.float32)

        return out

    def __repr__(self) -> str:
        msg = f" {self.__class__.__name__}"
        msg += f" split: {self.split}  max_size: {self.max_size}"
        msg += f" num_images: {self.__len__()}"
        return msg
