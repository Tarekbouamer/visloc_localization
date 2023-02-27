from typing import Any, Union, Dict, List

from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from loc.utils.transforms import normalize_img_net, to_gray

# logging
import logging
logger = logging.getLogger("loc")

class FeaturesExtractor:
    def __init__(self, 
                 cfg:Dict=None
                 ) -> None:
        #
        self.cfg    = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                       
    def _set_device(self) -> None:      
        self.extractor.to(device=self.device)              
    
    def _eval(self) -> None:
        self.extractor.eval()
            
    def _load_weights(self) -> None:     
        raise NotImplementedError
    
    def _normalize_imagenet(self, 
                            x: torch.Tensor
                            ) -> torch.Tensor:
        return normalize_img_net(x)

    def _to_gray(self, 
                 x: torch.Tensor
                ) -> torch.Tensor:
        return to_gray(x)         
    
    def _prepare_input_data(self, 
                        data: Dict,
                        **kwargs
                        ) -> Dict:
       # 
        __to_gray__     =  kwargs.pop("gray", False)
        __normalize__   =  kwargs.pop("normalize", False)

        #
        assert len(data["img"].shape) > 2, len(data["img"].shape)
        # bached
        if len(data["img"].shape) < 4:
            data["img"] = data["img"].unsqueeze(0)

        # to device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device=self.device)
                
        # gray
        if __to_gray__:
            data['img'] = self._to_gray(data['img'])
                
        # normalize
        if __normalize__:
            data['img'] = self._normalize_imagenet(data['img'])                 

        return data

    def _dataloader(self, 
                    _iter: Any
                    )-> DataLoader:
        
        if isinstance(_iter, DataLoader):
            return _iter
        else:
            return DataLoader(_iter, num_workers=self.cfg.num_workers)
                
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

            
            
            
     

        
  