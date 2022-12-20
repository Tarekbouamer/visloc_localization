import torch
from .base import BaseMatcher

# logger
import logging
logger = logging.getLogger("loc")


 
class MutualNearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='mutual_nearest_neighbor'):
        super().__init__(matcher_type)
        self.eval()
    
    @torch.no_grad()  
    def forward(self, desc1, desc2, distance_thresh=0.7, cross_check=True):
        """"""
        desc1 = desc1.squeeze(0) 
        desc2 = desc2.squeeze(0) 

        assert desc1.shape[1] != 0 and desc2.shape[1] != 0
        assert desc1.shape[0] == desc2.shape[0]

        nn_sim  = torch.einsum('dn, dm->nm', desc1, desc2)

        nn_dist_12, nn_idx_12 = torch.max(nn_sim,   dim=1)
        nn_dist_21, nn_idx_21 = torch.max(nn_sim,   dim=0)
        
        nn_dist = 2 * (1 - nn_dist_12)
        
        #
        ids1 = torch.arange(0, nn_sim.shape[0], device=desc1.device)
        
        # cross check
        mask = ids1 == nn_idx_21[nn_idx_12]
        
        # THD
        if distance_thresh:
            mask = mask & (nn_dist <= distance_thresh**2)
        
        matches = torch.where(mask, nn_idx_12,          nn_idx_12.new_tensor(-1))
        scores  = torch.where(mask, (nn_dist+1)/2.0,    nn_dist.new_tensor(0.))        

        return scores, matches

