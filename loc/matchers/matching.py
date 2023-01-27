# logger
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.functional as f

from . import BaseMatcher, MutualNearestNeighbor, SuperGlueMatcher

logger = logging.getLogger("VPS")


def make_matcher(cfg):
    
    matcher_name = cfg.matcher.name
    
    if matcher_name == "nn":
        return MutualNearestNeighbor(cfg.matcher)
    elif matcher_name == "superglue":
        return SuperGlueMatcher(cfg.matcher)
    else:
      raise KeyError(matcher_name)


class Matcher(BaseMatcher):
  
    def __init__(self, 
                 cfg:Dict={}
                 ) -> None:   
        super().__init__(cfg=cfg)
        
        # call from a factory
        self.matcher = make_matcher(cfg=cfg)
            
        # init
        self._set_device()
        self._eval()
    
    def _prepare_inputs(self,
                        data: Dict[str, Union[torch.Tensor, List, Tuple]]
                        ) -> Dict:
        raise NotImplementedError
 
    def _good_matches(self, 
                      keypoints0:torch.Tensor, 
                      keypoints1:torch.Tensor, 
                      matches0:torch.BoolTensor, 
                      matching_scores0:torch.FloatTensor, 
                      **kwargs
                      )-> Dict:
        raise NotImplementedError
      
    @torch.no_grad()
    def match_pair(self, 
                   data: Dict[str, Union[torch.Tensor, List, Tuple]]
                   ) -> Dict:
    
        preds =  self.matcher(data)
    
        return preds