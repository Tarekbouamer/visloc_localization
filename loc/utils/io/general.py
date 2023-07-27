# logger
from loguru import logger
import pickle
from pathlib import Path
# typing
from typing import Any, Dict, List, Tuple, Union

from loguru import logger 

def dump_logs(logs: Any, 
              save_path:Path
              )-> None:
    """log localization process into pkl file

    Args:
        logs (Any): bag of localization logs
        save_path (Path): save path
    """    

    save_path = save_path / 'visloc.logs.pkl'

    logger.info(f'writing logs to {save_path} ')
    
    with open(save_path, 'wb') as f:
        pickle.dump(logs, f)

        
def write_poses_txt(poses: List[Any], 
                    save_path: Path
                    ) -> None:
    """write camera poses to text files name qvec tvec

    Args:
        poses (List[Any]): list of camera poses
        save_path (Path): save path
    """    
    save_path = save_path / 'visloc.poses.txt'

    logger.info(f'writing poses to {save_path}...')

    with open(save_path, 'w') as tfile:
        
        for q in poses:
    
            name        = q.split('/')[-1]
            qvec, tvec  = poses[q]
            qvec        = ' '.join(map(str, qvec))
            tvec        = ' '.join(map(str, tvec))

            tfile.write(f'{name} {qvec} {tvec}\n')
    # 
    tfile.close()
        