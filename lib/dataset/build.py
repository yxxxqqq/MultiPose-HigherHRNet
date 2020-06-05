# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .HIEDataset import HIEDataset as hie
from .HIEKeypoints import HIEKeypoints as hie_kpt
from .HIECOCODataset import HIECOCODataset as hiecoco
from .HIECOCOKeypoints import HIECOCOKeypoints as hiecoco_kpt
from .HIEDataProcess import dataset_split
from .transforms import build_transforms
from .target_generators import HeatmapGenerator
from .target_generators import ScaleAwareHeatmapGenerator
from .target_generators import JointsGenerator


def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)   # scale, rotation, flip

    if cfg.DATASET.SCALE_AWARE_SIGMA:
        _HeatmapGenerator = ScaleAwareHeatmapGenerator
    else:
        _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS, cfg.DATASET.SIGMA
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]
    joints_generator = [
        JointsGenerator(
            cfg.DATASET.MAX_NUM_PEOPLE,
            cfg.DATASET.NUM_JOINTS,
            output_size,
            cfg.MODEL.TAG_PER_JOINT
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    if 'hie_kpt' == cfg.DATASET.DATASET:
        full_dataset = os.listdir(os.path.join(cfg.DATASET.ROOT, 'images'))
        train_dataset, valid_dataset = dataset_split(full_dataset)
        if is_train:
            dataset_name = train_dataset
        else:
            dataset_name = valid_dataset
        dataset = eval(cfg.DATASET.DATASET)(
            cfg,
            dataset_name,
            # is_train,
            heatmap_generator,
            joints_generator,
            transforms
        )
    else:
        dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
        dataset = eval(cfg.DATASET.DATASET)(
            cfg,
            dataset_name,
            is_train,
            heatmap_generator,
            joints_generator,
            transforms
        )

    return dataset


def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    transforms = None

    if 'hie' == cfg.DATASET.DATASET_TEST:
        full_dataset = os.listdir(os.path.join(cfg.DATASET.ROOT, 'images'))
        train_dataset, valid_dataset = dataset_split(full_dataset)
        dataset_name = valid_dataset
        dataset = eval(cfg.DATASET.DATASET_TEST)(
            cfg,
            dataset_name,
            transforms
        )
    else:
        dataset = eval(cfg.DATASET.DATASET_TEST)(
            cfg.DATASET.ROOT,
            cfg.DATASET.TEST,
            cfg.DATASET.DATA_FORMAT,
            transforms
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset
