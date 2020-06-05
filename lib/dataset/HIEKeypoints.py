#-*-coding:utf-8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from .HIEDataset import HIEDataset

logger=logging.getLogger(__name__)

class HIEKeypoints(HIEDataset):
    def __init__(self, cfg, dataset_name, heatmap_generator, joints_generator, transforms=None):
        super(HIEKeypoints, self).__init__(cfg.DATASET.ROOT, dataset_name)

        self.num_scales = self._init_check(heatmap_generator, joints_generator)

        self.num_joints=cfg.DATASET.NUM_JOINTS
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.scale_aware_sigma = cfg.DATASET.SCALE_AWARE_SIGMA
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.int_sigma = cfg.DATASET.INT_SIGMA

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator

    def __getitem__(self, index):
        img, anno = super().__getitem__(index)
        mask=self.get_mask(img)

        # to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        joints=self.get_joints(anno)
        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list=self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list


    def get_joints(self, anno):
        num_people=len(anno)

        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints'])[:, 1:4].reshape([-1, 3])
            if self.with_center:
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
            if self.scale_aware_sigma:
                # get person box
                box = obj['bbox']  # [x1, y1, x2, y2, score]
                size=max(box[2]-box[0], box[3]-box[1])  # width, height
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.round(sigma + 0.5))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma
        # print(joints.shape)
        return joints

    def get_mask(self, image):
        height=image.shape[0]
        width=image.shape[1]
        m=np.zeros((height, width))
        return m<0.5


    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(joints_generator)
            )
        return len(heatmap_generator)