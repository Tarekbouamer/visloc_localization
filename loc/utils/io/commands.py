import subprocess
from typing import List


def run_command(cmd: str, 
                args: List[str]
                ) -> None:
    """execute commands with arguments

    Args:
        cmd (str): command executable 
        args (List[str]): list of arguments
    """  

    # merge
    args.insert(0, cmd)

    # run
    subprocess.run(args, check=True)