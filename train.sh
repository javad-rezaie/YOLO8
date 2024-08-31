#
# Created on Sun Mar 31 2024
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
#MODEL_PATH="https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_s_mask-refine_syncbn_fast_8xb16-500e_coco_20230216_095938-ce3c1b3f.pth"
MODEL_PATH="https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco_20230217_120100-5881dec4.pth"
GPUS=3
# You can use the variables DATA_DIR, OUT_DIR, CONFIG_DIR, GPUS in your script from this point onward.

docker run -it --rm \
    --gpus all \
    --mount type=bind,source=$CONFIG_DIR,target=/configs \
    --mount type=bind,source=$DATA_DIR,target=/data \
    --mount type=bind,source=$OUT_DIR,target=/out \
    --shm-size 8g \
    mmyolo:latest \
    torchrun --nnodes 1 --nproc_per_node=$GPUS  /configs/main_train_mmengine.py /configs/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py $MODEL_PATH