
clear
python3 setup.py install


WORKSPACE='/media/dl/Data/datasets/aachen'
# WORKSPACE='/media/loc/D0AE6539AE65196C/VisualLocalization2020/aachen'

exhaustive_matching.py --workspace $WORKSPACE --mode loc
