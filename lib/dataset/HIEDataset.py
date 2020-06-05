from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import json
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from .HIEDataProcess import dataset_split

logger = logging.getLogger(__name__)


class HIEDataset(Dataset):
    '''
    "keypoint":{
        0: "nose",
        1: "chest",
        2: "right_shoulder",
        3: "right_elbow",
        4: "right_wrist",
        5: "left_shoulder",
        6: "left_elbow",
        7: "left_wrist",
        8: "right_hip",
        9: "right_knee",
        10: "right_ankle",
        11: "left_hip",
        12: "left_knee",
        13: "left_ankle",
    }

    Args:
        root(string): Root directory where dataset is located to.
        dataset(string):Dataset name (train2017, val2017, HIE20)
        data_format(string):Data format for reading('jpg', 'zip')
        transform(callable, optional):A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    '''

    def __init__(self, root, dataset, transform=None, target_transform=None):
        self.name = 'HIE'
        self.root = root
        self.dataset = dataset
        self.anno = self._get_anno_file_name()
        self.ids = list(range(0, len(self.dataset)))
        self.transform = transform
        self.target_transform = target_transform

        self.classes=['__background__', 'person']
        logger.info('=> classes:{}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))


    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        anno_path = os.path.join(self.root, 'annotations', 'person_keypoints.json')
        json_file = open(anno_path, encoding='utf-8').read()
        json_dict = json.loads(json_file)
        return json_dict

    def _get_anno_index(self, file_name):
        for i in range(len(self.anno)):
            if self.anno[i]['file_name'] == file_name:
                return i
        else:
            return -1

    def _get_image_path(self, file_name):
        images_dir=os.path.join(self.root, 'images')
        return os.path.join(images_dir, file_name)

    def __getitem__(self, index):
        '''
        Args:
            index(int): Index
        Return:
            tuple: Tuple (image, target).
        '''
        img_id = self.ids[index]
        file_name = self.dataset[img_id]
        anno_id = self._get_anno_index(file_name)

        assert anno_id != -1, 'There is no image annotation!'
        target = self.anno[anno_id]['persons']

        img=cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.ids)


    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp


    def evalue(self, cfg, preds, scores, output_dir, *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args:
        :param kwargs:
        :return:
        '''
        res_folder=os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file=os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.dataset
        )

        # preds is a list of: image x person x (keypoints)
        # keypoints: num_joints * 4 (x, y, score, tag)
        kpts=defaultdict(list)
        for idx, _kpts in enumerate(preds):
            img_id=self.ids[idx]
            file_name=self.dataset[img_id]
            for idx_kpt, kpt in enumerate(_kpts):
                area=(np.max(kpt[:, 0])-np.min(kpt[:, 0]))*(np.max(kpt[:, 1])-np.min(kpt[:, 1]))
                kpt=self.processKeypoints(kpt)
                # if self.with_center
                if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
                    kpt=kpt[:-1]

                kpts[int(file_name[-16:-4])].append({
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx][idx_kpt],
                    'tags': kpt[:, 3],
                    'image': int(file_name[-16:-4]),
                    'area':area
                })

        # rescoring and oks nms
        oks_nmsed_kpts=[]
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts=kpts[img]
            # if do not use nms, keep all detections
            keep=[]
            if len(keep)==0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        # HIE 'valid' set has annotation, 'test' has no annotation
        # calculate AP only on 'valid' set
        if args.test:
            return {'Null': 0}, 0
        else:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']


    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack=[
            {
                'cat_id': cls,
                'cls_ind': cls_ind,
                'cls': cls,
                'anno_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls=='__background__'
        ]

        results=self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s'%res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content=[]
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1]=']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)


    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id=data_pack['cat_id']
        keypoints=data_pack['keypoints']
        cat_results=[]
        num_joints=14

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points=np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points=np.zeros(
                (_key_points.shape[0], num_joints*3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt*3+0]=_key_points[:, ipt, 0]
                key_points[:, ipt*3+1]=_key_points[:, ipt, 1]
                key_points[:, ipt*3+2]=_key_points[:, ipt, 2]   # confidence score

            for k in range(len(img_kpts)):
                kpt=key_points[k].reshape((num_joints, 3))
                left_top=np.amin(kpt, axis=0)
                right_bottom=np.amax(kpt, axis=0)

                w=right_bottom[0]-left_top[0]
                h=right_bottom[1]-left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],   # coco format
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })
        return cat_results


    def _do_python_keypoint_eval(self, res_file, res_folder):
        pass










if __name__ == '__main__':
    root = '../../data/HIE20/'

    full_dataset=os.listdir(os.path.join(root, 'images'))
    train_dataset, valid_dataset=dataset_split(full_dataset)

    hie=HIEDataset(root, train_dataset)

    print(len(train_dataset))
    print(len(hie))

    for img, target in hie:
        print(img.shape)
    #     print(target)
