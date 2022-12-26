
clear
python3 setup.py install

clear

# DATA_DIR='/media/dl/Data/datasets/aachen/'
DATA_DIR='/media/loc/D0AE6539AE65196C/VisualLocalization2020/aachen/'

python3 -m loc.app \
      --directory $DATA_DIR \
      --config loc/configurations/default.yml
