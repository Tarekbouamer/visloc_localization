from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from core.io import H5Writer


class FeaturesWriter(H5Writer):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)

    def write_item(self,
                   key: str,
                   data: Tuple[torch.Tensor, np.ndarray],
                   name: str = "desc",
                   ) -> None:
        """write item to h5py disk
        Args:
            key (str): h5py key
            data (Tuple[torch.Tensor, np.ndarray]): array to save
            name (str, optional): name the array. Defaults to "desc".
            dtype (np.dtype, optional): convert to specific data type if not None
        """

        try:
            # create new group
            if key in self.hfile:
                del self.hfile[key]
            
            grp = self.hfile.create_group(key)
          
            if isinstance(data, torch.Tensor):
                data = self._to_numpy(data)

            # insert 
            grp.create_dataset(name, data=data)

        except OSError as error:
            raise error

    def write_items(self,
                    key: str,
                    data: Dict[str, Tuple[torch.Tensor, np.ndarray]],
                    ) -> None:

        try:
            # create new group
            if key in self.hfile:
                del self.hfile[key]
            
            grp = self.hfile.create_group(key)

                
            # write dict items
            for k, v in data.items():
                
                # convert 
                if isinstance(v, torch.Tensor):
                    v = self._to_numpy(v)
                
                # insert 
                grp.create_dataset(k, data=v)
                    
        except OSError as error:
            raise error


class MatchesWriter(H5Writer):
    def __init__(self, save_path: Path) -> None:
        super().__init__(save_path)
        
    def write_matches(self, data):
        
        pair_key, preds = data
    
        # del
        if pair_key in self.hfile:
            del self.hfile[pair_key]

        # 
        g = self.hfile.create_group(pair_key)

        #
        for k , v in preds.items():
            
            if isinstance(v, torch.Tensor):
                v = self._to_numpy(v)
                
            g.create_dataset(k, data=v)
    