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
      
def _cdist(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    r"""Manual `torch.cdist` for M1."""
    if (not is_mps_tensor_safe(d1)) and (not is_mps_tensor_safe(d2)):
        return torch.cdist(d1, d2)
    d1_sq = (d1**2).sum(dim=1, keepdim=True)
    d2_sq = (d2**2).sum(dim=1, keepdim=True)
    dm = d1_sq.repeat(1, d2.size(0)) + d2_sq.repeat(1, d1.size(0)).t() - 2.0 * d1 @ d2.t()
    dm = dm.clamp(min=0.0).sqrt()
    return dm


def match_snn(desc1, desc2, th=0.7):

    distance_matrix = torch.cdist(desc1, desc2)
    vals, idxs_in_2 = torch.topk(distance_matrix, 2, dim=1, largest=False)
    
    ratio = vals[:, 0] / vals[:, 1]
    
    mask = ratio <= th
    match_dists = ratio[mask]
    
    idxs_in1        = torch.arange(0, idxs_in_2.size(0), device=distance_matrix.device)[mask]
    idxs_in_2       = idxs_in_2[:, 0][mask]
    matches_idxs    = torch.cat([idxs_in1.view(-1, 1), idxs_in_2.view(-1, 1)], dim=1)
    
    #
    match_dists     = match_dists.view(-1, 1)
    matches_idxs    = matches_idxs.view(-1, 2)
    
    return match_dists, matches_idxs

      
# class MutualNearestNeighbor(BaseMatcher):
#     def __init__(self, matcher_type='mutual_nearest_neighbor'):
#         super().__init__(matcher_type)
#         self.eval()
    
#     @torch.no_grad()  
#     def forward(self, desc1, desc2, distance_thresh=0.7, cross_check=True):
#         """"""
#         #
        
#         sim         = torch.matmul(desc1, desc2.t())
#         sim1, indx1 = sim.topk(2, dim=1, largest=True)
#         dist1       = torch.sqrt(2 * (1 - sim1.clamp(-1, 1)))

#         mask        = torch.ones(dist1.shape[0], dtype=torch.bool, device=sim.device)
        
#         #
#         sim1, dist1, indx1 = sim1[:, 0], dist1[:, 0], indx1[:, 0]
        
#         if distance_thresh:
#             mask = mask & (dist1 < distance_thresh)
        
#         if cross_check:
#             _, indx2 = sim.topk(1, dim=0, largest=True)
#             mask = mask & (torch.arange(ind1.shape[0]) == indx2[0, ind1])

#         ind0        =  torch.arange(ind1.shape[0])[mask].to(torch.long)
#         ind1, sims  = ind1[mask].to(torch.long), sims[mask].to(torch.float)
#         matches     = torch.stack([ind0, ind1], dim=1).permute([1, 0])
        
#         print(matches.shape)
#         print(sims.shape)
#         return sims, matches
#         # cost = torch.cdist(desc1, desc2)
#         #     match_snn
#         # # 
#         # match_dists1, idxs_in_1 = torch.min(cost, dim=0)
#         # match_dists2, idxs_in_2 = torch.min(cost.t(), dim=0)
        
#         # _idxs   = torch.arange(cost.size(0), device=cost.device)
#         # mutuals = _idxs == idxs_in_1[idxs_in_2]
        
#         # if distance_thresh:
#         #     mutuals = mutuals & (match_dists2 <= distance_thresh)
        
#         # matches_idxs = torch.where(mutuals, idxs_in_2, idxs_in_2.new_tensor(-1))
#         # match_dists  = torch.where(mutuals, (match_dists2 + 1) / 2 , idxs_in_2.new_tensor(0))
    
#         # return match_dists, matches_idxs



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
      