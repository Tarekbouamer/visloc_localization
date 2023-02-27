import subprocess
from loc.utils.logging import setup_logger

def run_cmd(cmd, args):

  #
  args.insert(0, cmd)  
  
  logger.debug(f"execute {args}")

  #
  subprocess.run(args)


if __name__ == '__main__':
  
  logger = setup_logger(output=".", name="loc")

  
  WORKSPACE ='/media/loc/D0AE6539AE65196C/VisualLocalization2020/aachen'

  args = [
        "--workspace", WORKSPACE,
        "--dataset", "aachen",
        "--split", "db"
        ]
  
  run_cmd(cmd="image_retrieval.py", args=args)