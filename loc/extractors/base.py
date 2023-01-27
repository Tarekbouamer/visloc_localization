from typing import Any, Union, Dict, List

from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# logging
import logging
logger = logging.getLogger("loc")


class BaseExtractor:
    def __init__(self, 
                 name:str,
                 model:Union[str, nn.Module]=None, 
                 cfg:Dict=None
                 ) -> None:
      
        self.name   = name
        self.model  = model
        self.cfg    = cfg
               
    def _set_device(self) -> None:      
        self.model.to(device=self.device)              
    
    def _eval(self) -> None:
        self.model.eval()
            
    def _load_weights(self) -> None:     
        raise NotImplementedError
    
    def _normalize_imagenet(self, 
                            x: torch.Tensor
                            ) -> torch.Tensor:
        return self.transform(x)
         
    def _prepare_inputs(self, 
                        x: torch.Tensor
                        ) -> torch.Tensor:
        # bached
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        
        # device
        x = x.to(device=self.device) 

        return x
            
    @torch.no_grad()     
    def extract_image(self, 
                      img: torch.Tensor, 
                      name: str, 
                      scales: List=[1.0]
                      ) -> dict:
        """extract an image features
        Args:
            img (torch.Tensor): image tensor
            name (str): image name
            scales (list, optional): extraction sclaes. Defaults to [1.0].
        Returns:
            dict: network output data
        """  
        raise NotImplementedError

    @torch.no_grad()     
    def extract_dataset(self, 
                        dataset: Union[Dataset, DataLoader], 
                        scales:List=[1.0], 
                        save_path:Path=None
                        ) -> Dict:
        """extract dataset image features
        Args:
            dataset (Dataset): images dataset
            scales (list, optional): list of extraction scales. Defaults to [1.0].
            save_path (Path, optional): save path. Defaults to None.
            
        Returns:
            dict: extraction output
        """ 
        raise NotImplementedError
            
            
            
     

        
  