
# General
import sys
from collections import OrderedDict
import argparse
from os import mkdir
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(device=device) 
        print(device)
    
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

    def __to_numpy__(self, preds):
        out = {}
        for k, v in preds.items():
            if v[0].is_cuda:
                out[k] = v[0].cpu()
            else:
                out[k] = v[0].cpu().numpy()
        
        return out            

    def __write__(self, name, data):   
        hf  = self.writer
        
        try:
            if name in hf:
                del hf[name]
            
            g = hf.create_group(name)

            # write items
            for k, v in data.items():
                g.create_dataset(k, data=v)

        except OSError as error:   
            raise error 
        
    @torch.no_grad()      
    def extract_keypoints(self, dataset, save_path=None, **kwargs):
        
        #
        self.net.eval()

        #
        if save_path is not None:
            if kwargs.pop('override', False):
                return {'save_path': save_path}
            else:
                # writer
                self.writer = h5py.File(str(save_path), 'a')
        
        # dataloader
        dataloader = self.__dataloader__(dataset)

        # run --> 
        for it, data in enumerate(tqdm(dataloader, total=len(dataloader), colour='cyan', desc='extract keypoints'.rjust(15))):
            
            img = data['img']
            
            # prepare inputs
            img  = self.__prepare_input__(img, **kwargs) 
            img  = self.__cuda__(img) 
            
            # extract locals (W, H) order
            preds = self.net({'image': img})
            preds = self.__to_numpy__(preds)
            
            #
            preds['size'] = original_size = data['size'][0].numpy()
            
            # scale keypoints to original scale
            current_size    = np.array(img.shape[-2:][::-1])
            scales          = (original_size / current_size).astype(np.float32)
            preds['keypoints'] = (preds['keypoints'] + .5) * scales[None] - .5
            
            # write
            if hasattr(self, 'writer'):
                name = data['name'][0]
                self.__write__(name, preds)
                
        # 
        out = {
            'save_path': save_path
        }
        
        return out

        