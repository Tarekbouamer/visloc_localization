from typing import Any, Union, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# logger
import logging
from loguru import logger

class BaseMatcher:
    """Base Matcher class
    """    
    def __init__(self, 
                 cfg:Dict
                 ) -> None:
        """

        Args:
            cfg (Dict): config params
        """        
        super().__init__()

        # cfg
        self.cfg    = cfg
        
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _set_device(self):   
        """set to cuda if exsists
        """        
        self.matcher.to(device=self.device)              
    
    def _eval(self):     
        """set model to eval if matcher is nn.Module
        """  
        if isinstance(self.matcher, nn.Module):   
            self.matcher.eval()
            
    def _load_weights(self):
        pass
        
    @torch.no_grad()     
    def match_pair(self,
                   data: Dict[str, Union[torch.Tensor, List, Tuple]],
                    ) -> Dict: 
        """_summary_

        Args:
            data (Dict[str, Union[torch.Tensor, List, Tuple]]): input data of pair of images 

        Returns:
            Dict: pair predictions {matches and scores }
        """
        assert "descriptor0" in data.keys, KeyError("descriptor0 missing")
        assert "descriptor1" in data.keys, KeyError("descriptor1 missing")   
        
        raise NotImplementedError

    @torch.no_grad()     
    def match_sequence(self, 
                       data: Dict[str, Union[torch.Tensor, List, Tuple]],
                        ) -> Dict:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        msg  = f" {self.__class__.__name__}"
        msg += f" ("
        msg += f" model_name: {self.model_name} device: {self.device}"
        msg += f" cross_check: {self.cfg.matcher.cross_check}"
        msg += f" good_matches: {self.cfg.matcher.good_matches}"
        msg += f" )"
        return msg