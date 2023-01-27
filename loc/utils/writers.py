from typing import Dict, Tuple
import h5py
from pathlib import Path
import torch
import numpy as np

# logger
import logging
logger = logging.getLogger("VPS")


class VPSWriter:
    def __init__(self, 
                 save_path: Path
                 ) -> None:
        
        # save file
        self.save_path = save_path
        
        # writer
        self.file = h5py.File(str(save_path), 'a')

    def close(self):
        self.file.close()  
        
    def _to_numpy(self, 
                  x:torch.Tensor
                  ) -> torch.Tensor:
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.cpu()
        
    def write_item(self, 
                   key: str,
                   data: Tuple[torch.Tensor, np.ndarray],
                   name: str ="desc"
                   ) -> None:
        """write item to h5py disk
        Args:
            key (str): h5py key
            data (Tuple[torch.Tensor, np.ndarray]): array to save
            name (str, optional): name the array. Defaults to "desc".
        """        
        
        try:
            if key in self.file:
                del self.file[key]
            
            g = self.file.create_group(key)
                
            if not isinstance(data, np.ndarray):
                data = self._to_numpy(data)

            # TODO: half data 32/16/8 
             
            g.create_dataset(name, data=data)

        except OSError as error:   
            raise error  
        
        
    def write_items(self, 
                   key: str,
                   data: Dict[str, Tuple[torch.Tensor, np.ndarray]]
                   ) -> None:   
                
        try:
            if key in self.file:
                del self.file[key]
            
            g = self.file.create_group(key)

            for k, v in data.items():
                
                if not isinstance(v, np.ndarray):
                    v = self._to_numpy(v)
                    
                # TODO: half data 32/16/8 
                
                g.create_dataset(k, data=v)

        except OSError as error:   
            raise error 