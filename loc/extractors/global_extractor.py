# logging
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from retrieval.models import create_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from loc.utils.writers import FeaturesWriter

from .base import FeaturesExtractor

from loguru import logger
class GlobalExtractor(FeaturesExtractor):
    def __init__(self,
                 cfg: Dict = None
                 ) -> None:
        super().__init__(cfg)
      
        #
        model_name  = self.cfg.retrieval.model_name
        self.model_name = model_name

        self.extractor  = create_model(model_name=model_name, pretrained=True)

        # init
        self._set_device()
        self._eval()
            
    @torch.no_grad()     
    def extract_image(self,
                      data: Dict,
                      scales: List=[1.0],
                        **kwargs 
                        ) -> dict:          
        
        # prepare inputs
        data  = self._prepare_input_data(data, **kwargs)
        
        # extract
        preds = self.extractor.extract_global(data['img'], scales=scales, do_whitening=True)  
            
        # 
        preds["features"]  = preds["features"][0]  
        
        #        
        return preds

    @torch.no_grad()     
    def extract_dataset(self, 
                        dataset: Union[Dataset, DataLoader], 
                        scales:List=[1.0], 
                        save_path:Path=None,
                        **kwargs
                        ) -> Dict:
        # features writer 
        self.writer = FeaturesWriter(save_path)
            
        # dataloader
        _dataloader = self._dataloader(dataset)
        
        # time
        start_time = time.time()
                
        # run --> 
        for it, data in enumerate(tqdm(_dataloader, total=len(_dataloader), colour='green', desc='extract global'.rjust(15))):
            
            #
            it_name = data['name'][0] 
                       
            # extract
            preds = self.extract_image(data, scales, **kwargs)

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
       
        return save_path

    def __repr__(self) -> str:
        msg  = f"{self.__class__.__name__}"
        msg += f" ("
        msg += f" model_name: {self.model_name} "
        msg += f" device: {self.device} "
        msg += f" max_size: {self.cfg.retrieval.max_size}"
        msg += f")"
        return msg
            
              
            
            
     

        
  