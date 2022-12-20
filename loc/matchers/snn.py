import torch
from .base import BaseMatcher



def find_nn(sim, ratio_thresh=None, distance_thresh=None):
    

    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)

    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)

    # if ratio_thresh:
    #     mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
        
    matches = torch.where(mask,     ind_nn[..., 0],     ind_nn.new_tensor(-1))
    scores = torch.where(mask,  (sim_nn[..., 0]+1)/2,   sim_nn.new_tensor(0))

    return matches, scores


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class SNearestNeighbor(BaseMatcher):
    def __init__(self, matcher_type='nearest_neighbor'):
        super().__init__(matcher_type)
        
    def forward(self, desc1, desc2):
      
        sim = torch.einsum('nd,md->nm', desc1, desc2)
        
        ratio_thresh =  None
        distance_thresh = 0.7
        
        matches1, dist1 = find_nn(sim, 
                                  ratio_thresh=ratio_thresh,
                                  distance_thresh=distance_thresh)
        
        matches2, dist2 = find_nn(sim.transpose(0, 1),
                                  ratio_thresh=ratio_thresh,
                                  distance_thresh=distance_thresh)
        
        matches = mutual_check(matches1, matches2)
        
        return dist1, matches