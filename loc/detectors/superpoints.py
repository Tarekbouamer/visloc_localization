
# General
import sys
from collections import OrderedDict
import argparse
from os import mkdir
import numpy as np
from pathlib import Path
from tqdm import tqdm

# torch
import torch
from torch.utils.data import DataLoader

# third
from thirdparty.SuperGluePretrainedNetwork.models.superpoint import SuperPoint as superpoint


class SuperPoint(torch.nn.Module):
    """ Image Matching Frontend SuperPoint """
    def __init__(self, config={}):
        super().__init__()
        
        self.net = superpoint(config)

    def _forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
                
        return pred
    
    def __dataloader__(self, x, **kwargs):
        if isinstance(x, DataLoader):
            return x
        else:
            return DataLoader(x, num_workers=1, shuffle=False, drop_last=False, pin_memory=True)   


    def __prepare_input__(self, x, **kwargs):
        
        # BCHW
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        
        return x

    def __cuda__(self, x):
        if torch.cuda.is_available():
            return x.cuda()
        else: # cpu
            return x
                     
    def extract_keypoints(self, dataset, **kwargs):
        
        # dataloader
        dataloader = self.__dataloader__(dataset)

        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='cyan', desc='extract keypoints'.rjust(15))):
            
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img, **kwargs) 
            img  = self.__cuda__(img) 
            
            # extract locals 
            pred = self.net({'image': img})
            print(pred)
            input()

        