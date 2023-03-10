from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

from features.factory import create_local_feature, load_pretrained
from features.register import register_local_feature, get_pretrained_cfg, list_modules
from features.models.model_factory import create_model


class LocalFeature(nn.Module):

    def __init__(self, detector: nn.Module, descriptor: nn.Module) -> None:
        super().__init__()
        self.detector = detector
        self.descriptor = descriptor

    def forward(self, data: Dict) -> Dict:
        """detect & computer

        Args:
            data (Dict): _description_

        Returns:
            Dict: _description_
        """        
        preds, x = self.detector.detect(data)
        
        preds = self.descriptor.compute({**data, **preds}, x)
        
        return preds
      
      
def _make_local_feature(detector_name, descriptor_name, cfg=None, **kwargs):
    #
    detector = create_model(detector_name, cfg=cfg)
    #
    descriptor = create_model(descriptor_name, cfg=cfg)
    
    model = LocalFeature(detector=detector, descriptor=descriptor)
    
    return model
    

@register_local_feature
def superpoint(cfg=None, **kwargs):
    return _make_local_feature(detector_name="superpoint", descriptor_name="superpoint", cfg=cfg)


if __name__ == '__main__':
    img = torch.rand([1, 1, 1024, 1024])
    model = create_local_feature("superpoint")
    print(model)

    preds = model({'image': img})
    
    for k, v in preds.items():
        print(k,"   ", v[0].shape)