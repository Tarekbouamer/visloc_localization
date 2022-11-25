
from typing import List
import numpy as np
from collections import OrderedDict

from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

# core
import core.backbones   as models
from core.utils.parallel import PackedSequence
from core.utils.PCA         import PCA, PCA_whitenlearn_shrinkage
from core.utils.snapshot    import save_snapshot, resume_from_snapshot, pre_train_from_snapshots

# image_ret
from image_retrieval.datasets.generic import ImagesFromList, ImagesTransform, INPUTS
from image_retrieval.modules.heads.head         import globalHead


class GF_Net(nn.Module):
    
    def __init__(self, body, ret_head, num_features=100):
        super(GF_Net, self).__init__()
        
        self.body           = body
        self.ret_head       = ret_head
        self.num_features   = num_features

    def extract_body_features(self, img, scales=[1.0]):
        
        if not isinstance(scales, List):
            raise TypeError('scales should be list of scale factors, example [0.5, 1.0, 2.0]')        
        
        #
        features, sizes = [], []
        for s in scales :
            
            s_img = nn.functional.interpolate(img, scale_factor=s, mode='bilinear', align_corners=False)
            
            # body
            x = self.body(s_img)
            
            # level
            if isinstance(x, dict):
                last_mod = next(reversed(x))
                x = x[last_mod] 
            
            # append           
            features.append(x)
            sizes.append(s_img.shape[-2:])
        
        return features, sizes

    def random_sampler(self, L):
    
        K           = min(self.num_features, L) if self.num_features is not None else L
        indices     = torch.randperm(L)[:K]  
        
        return indices  
                
    def extract_local_features(self, features, scales, do_whitening):
        """
            Extract local descriptors at each scale 
        """
        
        # list of decs 
        descs_list, locs_list = self.ret_head.forward_locals(features, scales, do_whitening)
        
        # ---> tensor
        descs = torch.cat(descs_list,   dim=1).squeeze(0)
        locs  = torch.cat(locs_list,    dim=1).squeeze(0)
        
        # sample 

        L   = descs.shape[0]
        idx = self.random_sampler(L)
                
        return descs[idx], locs[idx]
      
    def extract_global_features(self, features, scales, do_whitening):
        
        descs_list = self.ret_head.forward(features, scales,  do_whitening)

        # ---> tensor
        descs = torch.cat(descs_list,   dim=0)
                
        # sum and normalize
        descs   = torch.sum(descs, dim=0).unsqueeze(0)
        descs   = nn.functional.normalize(descs, dim=-1, p=2, eps=1e-6)
        
        return descs
        
    def forward(self, img=None, scales=[1.0],  do_whitening=False, do_local=False, **varargs):

        # Run network body
        features, img_sizes = self.extract_body_features(img, scales=scales)
        
        # Run head
        descs, locs =None, None
        
        if do_local:
            descs, locs = self.extract_local_features(features, scales, do_whitening)
        else:
            descs = self.extract_global_features(features, scales, do_whitening)
            
        pred = OrderedDict([
            ("descs", descs),
            ("locs", locs)
        ])

        return pred



def make_model(logger=None):
    
    # Body
    body_fn     = models.__dict__[default_config['arch']]
    body        = body_fn()
        
    # Head
    head = globalHead(  inp_dim=default_config['inp_dim'],
                        global_dim=default_config['global_dim'],
                        local_dim=default_config['local_dim'],
                        pooling=default_config['pooling'],
                        do_withening=default_config['do_withening'],
                        layer=default_config['layer'],
                        )

    # Create image retrieval network
    model = GF_Net(body, head, num_features=default_config['num_features'])
    
    # 
    if logger:
        logger.info("Loading snapshot from %s", default_config['weights_path'])
        snapshot = resume_from_snapshot(model, default_config['weights_path'], ["body", "ret_head"])  
    
    return model
    

default_config = {
        'weights_path':     "experiments/res4_50/test_model_best.pth.tar",
        
        'arch':             "resnet_4_50",
        'inp_dim':          1024,
        'global_dim':       1024,
        'local_dim':        128,
        'pooling':          {"name": "GeM", "params": {"p":3, "eps": 1e-6}},
        'do_withening':     True,
        'layer':            "linear",
        'num_features':     2000,

}
    
# Interface    
class GF_NetFeatures: 
    def __init__(self, do_cuda=True, logger=None):

        # Gpu stuff    
        self.cuda   = torch.cuda.is_available() & do_cuda
        self.device = torch.device('cuda' if self.cuda else 'cpu')
                
        # logger
        if logger:
            logger.info('GF_Net_Features')
            
        self.net = make_model(logger=logger);
        #
        if self.cuda:
            self.net.cuda(self.device)
        # 
        self.net.eval()
        
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                         std=(0.229, 0.224, 0.225)),
        ])
                                
    # compute both keypoints and descriptors       
    def __call__(self, img):
        
        # --> Pil  
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(img)
        
        # --> Tensor and normalize
        img = self.transform(img).unsqueeze(0)
        
        # --> device
        img = img.to(device=self.device)
        
        with torch.no_grad():
            # Run network body
            features, _ = self.net.extract_body_features(img)
            
            # Run head
            descs = self.net.extract_global_features(features, scales=[1.0], do_whitening=True)
            
        # Out
        pred = { 'global_descriptor': descs }
        
        return pred
