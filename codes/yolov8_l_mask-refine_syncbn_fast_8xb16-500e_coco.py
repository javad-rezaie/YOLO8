#
# Created on Sun Mar 31 2024
#
# Copyright (c) 2024 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#

# Import a base configuration file that serves as the foundation for your training setup. 
# This base configuration typically includes default settings for model architecture, optimizer, and other parameters.

_base_ = [
    'mmyolo::yolov8/yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py'
]

# dataset settings
# Adjust the paths within the configuration to point to your dataset directory. 
# This ensures that the training pipeline accesses the correct data during the training process.

data_root = "/data/" 
train_annot = "train_coco.json"
val_annot = "test_coco.json"
test_annot = "test_coco.json"
train_image_folder = "images/"
val_image_folder = "images/"
test_image_folder = "images/"

# Training Parameter Settings
# Specify the hyperparameters for training, such as batch size, number of epochs, learning rate, weight decay, etc. 
# These parameters significantly impact the training process and model performance.

base_lr = 0.0001
lr_factor = 0.1
max_epochs = 100
warmup_epochs = 5
check_point_interval = 10
val_interval =  1
close_mosaic_epochs  = 10
val_interval_stage2 = 1
batch_size = 4
work_dir = "/out"

# Modify the model configuration to accommodate the specific requirements of your dataset. 
# This might involve adjusting the input/output dimensions, changing the number of output classes, or fine-tuning certain layers to better suit your data (here we only need to update the number of objects).

train_data_annot_path = data_root + train_annot

def get_object_classes(path_to_annotation):
    import json
    with open(path_to_annotation, "r") as f:
        data = json.load(f)
    cats = [cat['name'] for cat in data["categories"]]
    return tuple(cats)


classes = get_object_classes(train_data_annot_path)

metainfo = {
    'classes': classes
}
num_classes = len(classes)


model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes = num_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes = num_classes)
    )
    )

# Customize the data loaders to preprocess and load your dataset efficiently. 
# This step involves setting up data augmentation techniques, data normalization, and any other preprocessing steps necessary for training.

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=3,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    collate_fn=dict(_delete_=True, type='yolov5_collate'),
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_annot,
        data_prefix=dict(img=train_image_folder)
        )
    )

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=3,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_annot,
        data_prefix=dict(img=val_image_folder)
        )
    )

test_dataloader = dict(
    batch_size=batch_size,
    num_workers=3,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_annot,
        data_prefix=dict(img=test_image_folder),
        test_mode=True
        )
    )

train_cfg = dict(
    val_interval=val_interval, 
    max_epochs=max_epochs,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)]
    )


val_evaluator = dict(
    ann_file=data_root + val_annot,
    )
test_evaluator = dict(
    ann_file=data_root + test_annot,
    )


# optimizer
# Configure the optimizer (e.g., SGD, Adam) and learning rate scheduler (e.g., step-based, cosine annealing) based on your training objectives and model architecture. 
# Tuning these components can significantly impact convergence speed and final model performance.


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_factor,
        by_epoch=True,
        begin=0,
        end=warmup_epochs),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=int(0.7*max_epochs),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        _delete_ = True,
        type='CheckpointHook',
        interval=check_point_interval, # Save checkpoint on "interval" epochs
        max_keep_ckpts=1  # only keep latest 1 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs, 
        switch_pipeline=_base_.train_pipeline_stage2
        )
]
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
# auto_scale_lr = dict(enable=True, base_batch_size=8*16)