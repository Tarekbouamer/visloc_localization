# logger
import logging
from typing import Dict, List, Tuple, Union

import torch
import torch.functional as f

from loc.matchers import BaseMatcher, MutualNearestNeighbor, SuperGlueMatcher

logger = logging.getLogger("loc")


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
        self._load_weights()
        
        # 
        logger.info(f"init {cfg.matcher.name} matcher")
    
    def _prepare_inputs(self,
                        data: Dict[str, Union[torch.Tensor, List, Tuple]]
                        ) -> Dict:
        
        for k, v  in data.items():
            # device
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device) 
            
        return data
 
    def _good_matches(self, 
                      preds: Dict[str, Union[torch.Tensor, List, Tuple]]
                      )-> Dict:
        raise NotImplementedError
      
    @torch.no_grad()
    def match_pair(self, 
                   data: Dict[str, Union[torch.Tensor, List, Tuple]]
                   ) -> Dict:
        
        assert "descriptors0" in data, KeyError("descriptors0 missing")
        assert "descriptors1" in data, KeyError("descriptors1 missing")
        
        # prepare
        data = self._prepare_inputs(data)

        # match
        preds =  self.matcher(data)
        
        # good matches
        if self.cfg.matcher.good_matches:
            preds = self._good_matches(preds)
    
        return preds