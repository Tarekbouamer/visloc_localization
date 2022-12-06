import torch
from .base import BaseMatcher

# logger
import logging
logger = logging.getLogger("loc")

class NearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='nearest_neighbor'):
        super().__init__(matcher_type)
    
    @ torch.no_grad()  
    def forward(self, desc1, desc2):
        """"""
        
        # sim
        cost = torch.cdist(desc1, desc2)
            
        # match distances 
        match_dists, idxs_in_2 = torch.min(cost, dim=1)

        mask = torch.ones(idxs_in_2.shape[0], dtype=torch.bool, device=idxs_in_2.device)
        
        matches_idxs    = torch.where(mask, idxs_in_2,      idxs_in_2.new_tensor(-1))
        match_dists     = torch.where(mask, match_dists,    idxs_in_2.new_tensor(0))

        # idxs_in1 = torch.arange(0, idxs_in_2.size(0), device=idxs_in_2.device)
        
        # match indices 
        # matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)

        # # 
        # match_dists  = match_dists.view(-1, 1)
        # matches_idxs = matches_idxs.view(-1, 2)

        return match_dists, matches_idxs
      
      
class MutualNearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='mutual_nearest_neighbor'):
        super().__init__(matcher_type)
    
    @ torch.no_grad()  
    def forward(self, desc1, desc2, distance_thresh=0.8):
        """"""
        #
        cost = torch.cdist(desc1, desc2)
            
        # 
        match_dists,  idxs_in_2 = torch.min(cost, dim=1)
        match_dists2, idxs_in_1 = torch.min(cost, dim=0)
        
        _idxs   = torch.arange(cost.size(0), device=cost.device)
        mutuals = _idxs == idxs_in_1[idxs_in_2]
        
        if distance_thresh:
            mutuals = mutuals & (match_dists <= distance_thresh**2)
        
        matches_idxs = torch.where(mutuals, idxs_in_2, idxs_in_2.new_tensor(-1))
        match_dists  = torch.where(mutuals, match_dists, idxs_in_2.new_tensor(0))
        
        
        # minsize_idxs = torch.arange(ms, device=cost.device)

        #
        # if cost.size(0) <= cost.size(1):
        #     mutual_nns    = minsize_idxs == idxs_in_1[idxs_in_2][:ms]
        #     matches_idxs  = torch.cat([minsize_idxs.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)[mutual_nns]
        #     match_dists   = match_dists[mutual_nns]
        # else:
        #     mutual_nns    = minsize_idxs == idxs_in_2[idxs_in_1][:ms]
        #     matches_idxs  = torch.cat([idxs_in_1.view(-1, 1), minsize_idxs.view(-1, 1)], dim=1)[mutual_nns]
        #     match_dists   = match_dists2[mutual_nns]
        
        return match_dists, matches_idxs

class NearestNeighborRatio(BaseMatcher):
    def __init__(self, matcher_type='nearest_neighbor_ratio', thd=0.):
        super().__init__(matcher_type)
        
        self.thd = thd
    
    @ torch.no_grad()  
    def forward(self, desc1, desc2):
        """"""
        cost = torch.cdist(desc1, desc2)
        
        #
        vals, idxs_in_2 = torch.topk(cost, 2, dim=1, largest=False)
        ratio   = vals[:, 0] / vals[:, 1]
        
        # mask
        mask    = ratio <= self.thd
        
        # distance 
        match_dists = ratio[mask]
        
        idxs_in1  = torch.arange(0, idxs_in_2.size(0), device=cost.device)[mask]
        idxs_in_2 = idxs_in_2[:, 0][mask]
        
        # matches 
        matches_idxs = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
        
        return match_dists.view(-1, 1), matches_idxs.view(-1, 2)
      