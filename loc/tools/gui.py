from typing import Any, Dict

from loc.utils.viewer import VisualizerGui


def visualizer_gui(args: Any,
                   cfg: Dict = None
                   ) -> None:
    """ visualizer gui

    Args:
        args (Any): arguments 
        cfg (Dict, optional): configuration. Defaults to None.
    """

    vis = VisualizerGui()
    vis.read_model(args.visloc_path)
    vis.create_window()
    vis.show()
