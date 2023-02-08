from typing import Dict, Tuple
import h5py
from pathlib import Path
import torch
import numpy as np

# logger
import logging
logger = logging.getLogger("loc")

class Writer:
    def __init__(self,
                 save_path: Path
                 ) -> None:
        # save file
        self.save_path = save_path

        # writer
        self.hfile = h5py.File(str(save_path), 'a')
        
    def close(self):
        self.hfile.close()
        
    def _to_numpy(self,
                  x: torch.Tensor
                  ) -> torch.Tensor:
        if x.is_cuda:
            return x.cpu().numpy()
        else:
            return x.numpy()
        
class FeaturesWriter(Writer):
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
            if key not in self.hfile:
                grp = self.hfile.create_group(key)
            else:
                grp = self.hfile[key]
           
            if isinstance(data, torch.Tensor):
                data = self._to_numpy(data)

            # insert 
            if name in grp:
                grp[name][...] = data
            else:
                grp.create_dataset(name, data=data)

        except OSError as error:
            raise error

    def write_items(self,
                    key: str,
                    data: Dict[str, Tuple[torch.Tensor, np.ndarray]],
                    ) -> None:

        try:
            # create new group
            if key not in self.hfile:
                grp = self.hfile.create_group(key)
            else:
                grp = self.hfile[key]
                
            # write dict items
            for k, v in data.items():
                # convert 
                if isinstance(v, torch.Tensor):
                    v = self._to_numpy(v)
                # insert 
                if k in grp:
                    grp[k][...] = v
                else:
                    grp.create_dataset(k, data=v)
                    
        except OSError as error:
            raise error


class MatchesWriter(Writer):
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
#TODO: add call function that calls write matches  
    