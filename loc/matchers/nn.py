from typing import Dict
import torch
import torch.nn as nn

# logger
import logging
logger = logging.getLogger("loc")
 
class MutualNearestNeighbor(nn.Module):
    """mutual nearest neighboor matcher
    """    
    def __init__(self, cfg:Dict={}):
        """_summary_

        Args:
            cfg (Dict, optional): configs. Defaults to {}.
        """        
        super().__init__()
        
        self.cfg = cfg
        
    
    @torch.no_grad()  
    def forward(self, data):
        
        desc0 = data["descriptors0"]
        desc1 = data["descriptors1"]

        if len(data["descriptors0"].shape) > 2:
            desc0 = desc0.squeeze(0) 
            desc1 = desc1.squeeze(0) 

        assert desc0.shape[1] != 0 and desc1.shape[1] != 0
        assert desc0.shape[0] == desc1.shape[0]

        nn_sim  = torch.einsum('dn, dm->nm', desc0, desc1)

        nn_dist_01, nn_idx_01 = torch.max(nn_sim,   dim=1)
        nn_dist_10, nn_idx_10 = torch.max(nn_sim,   dim=0)
        
        nn_dist = 2 * (1 - nn_dist_01)
        
        #
        ids1 = torch.arange(0, nn_sim.shape[0], device=desc0.device)
        
        # cross check
        mask = ids1 == nn_idx_10[nn_idx_01]
        
        # thd
        if self.cfg.distance_thresh> 0.0:
            mask = mask & (nn_dist <= self.cfg.distance_thresh**2)
        
        matches = torch.where(mask, nn_idx_01,          nn_idx_01.new_tensor(-1))
        scores  = torch.where(mask, (nn_dist+1)/2.0,    nn_dist.new_tensor(0.))        

        out = {
            "matches": matches,
            "scores": scores,
            }
        return out
    
    

