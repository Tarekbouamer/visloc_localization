from typing import Any, Union, Dict, List

from pathlib import Path

import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from .base import FeaturesExtractor
from retrieval.datasets import ImagesListDataset
from retrieval.models import create_model, get_pretrained_cfg
from retrieval.utils.logging import setup_logger

from loc.utils.writers import FeaturesWriter

# logging
import logging
logger = logging.getLogger("loc")


class GlobalExtractor(FeaturesExtractor):
    def __init__(self,
               cfg: Dict = None
               ) -> None:
        super().__init__(cfg)
      
        #
        model_name  = self.cfg.retrieval.model_name
        self.extractor  = create_model(model_name=model_name)

        # init
        self._set_device()
        self._eval()
            
    @torch.no_grad()     
    def extract_image(self, 
                        data: Dict, 
                        scales: List=[1.0],
                        **kwargs 
                        ) -> dict:          
        # 
        it_name = data['name']
            
        # normalize
        if kwargs.pop("normalize", False):
            data['img'] = self._normalize_imagenet(data['img'])                    
            
        # prepare inputs
        data  = self._prepare_inputs(data)
            
        # extract
        preds = self.extractor.extract_global(data['img'], scales=scales, do_whitening=True)    
        preds["features"] = preds["features"][0]
        
        out = {
          "features" :  torch.stack([preds["features"]]),
          "names" :     np.stack(it_name),
        }
        
        return out

    @torch.no_grad()     
    def extract_dataset(self, 
                        dataset: Union[Dataset, DataLoader], 
                        scales:List=[1.0], 
                        save_path:Path=None,
                        normalize: bool = False 
                        ) -> Dict:
        
        # features writer 
        self.writer = FeaturesWriter(save_path)
            
        # dataloader
        _dataloader = self._dataloader(dataset)

        # 
        features = []
        names = []
        
        # time
        start_time = time.time()
                
        # run --> 
        for it, data in enumerate(tqdm(_dataloader, total=len(_dataloader), colour='green', desc='extract global'.rjust(15))):
            
            #
            it_name = data['name'][0] 
            
            # normalize
            if normalize:
                data['img'] = self._normalize_imagenet(data['img'])                    
            
            # prepare inputs
            data  = self._prepare_inputs(data)
            
            # extract
            preds = self.extractor.extract_global(data['img'], scales=scales, do_whitening=True)    
            preds["features"] = preds["features"][0]

            #
            features.append(preds["features"])
            names.append(it_name)
            
            # write preds
            self.writer.write_items(key=it_name, data=preds)
            
            # clear cache  
            if it % 10 == 0:
                torch.cuda.empty_cache()
                    
        # close writer     
        self.writer.close()  
            
        # end time
        end_time = time.time() - start_time  
            
        logger.info(f'extraction done {end_time:.4} seconds saved {save_path}')
      
        #
        out = {
          "features" :  torch.stack(features),
          "names" :     np.stack(names),
          "save_path":  save_path
        }
        
        return out
            
              
            
            
     

        
  