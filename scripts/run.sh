
clear
python3 setup.py install

EXPERIMENT='./experiments/'
DATA_DIR='/media/dl/Data/datasets/'

python3 ./scripts/run.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      --config .visloc/configurations/defaults/default.ini \
      # --eval 
