from typing import Any, Union, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# logger
import logging
logger = logging.getLogger("loc")

class BaseMatcher:
    def __init__(self, 
                 cfg:Dict
                 ) -> None:
        super().__init__()
      
        self.cfg    = cfg

    def _set_device(self):   
        self.matcher.to(device=self.device)              
    
    def _eval(self):     
        self.matcher.eval()
            
    def _load_weights(self):
        raise NotImplementedError
        
    @torch.no_grad()     
    def match_pair(self,
                   data: Dict[str, Union[torch.Tensor, List, Tuple]],
                    ) -> Dict: 
        raise NotImplementedError

    @torch.no_grad()     
    def match_sequence(self, 
                       data: Dict[str, Union[torch.Tensor, List, Tuple]],
                        ) -> Dict:
        raise NotImplementedError