#
# Created on Sun Mar 10 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#

DATA_DIR="/mnt/SSD2/coco_stuff10k/" # In the container, it is acceable as /data
#Data Structure on my local computer is as follows:
#/mnt/SSD2/coco_stuff10k/
#├── images/
#├── train_coco.json
#└── test_coco.json

#It will be mapped to container, and it seems like:
#/data/
#├── images/
#├── train_coco.json
#└── test_coco.json

OUT_DIR="$PWD/out" # In the container, it is acceable as /out
CONFIG_DIR=$PWD/codes/ # In the container, it is acceable as /configs
NOTEBOOK_DIR=$PWD/notebooks/ # In the container, it is acceable as /notebooks

docker run -it --rm \
    --gpus all \
    --mount type=bind,source=$CONFIG_DIR,target=/configs \
    --mount type=bind,source=$DATA_DIR,target=/data \
    --mount type=bind,source=$OUT_DIR,target=/out \
    --mount type=bind,source=$NOTEBOOK_DIR,target=/notebooks \
    --shm-size 8g \
    -p 8888:8888 \
    mmyolo:latest \
    jupyter-lab  --ip 0.0.0.0 --port 8888 --allow-root /notebooks