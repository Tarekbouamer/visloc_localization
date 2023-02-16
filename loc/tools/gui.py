import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / '../'))
print(sys.path[-1])
input()
import logging
import argparse
from omegaconf import OmegaConf

from loc.utils.viewer import VisualizerGui

def visualizer_gui(args, cfg):
      
    vis = VisualizerGui()
    vis.read_model(args.visloc_path)
    vis.create_window()
    vis.show()