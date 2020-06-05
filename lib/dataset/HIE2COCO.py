#-*-coding:utf-8-*-
import numpy as np
import os
import shutil

import torch
import cv2
import json


class CreateCoco():
    def __init__(self, root, train_factor, split=False):
        '''
        root: root directory
        train_factory: train dataset ratio
        split: whether to separate full dataset to 'train' and 'valid' set
        '''
        self.root=root
        self.train_factor=train_factor
        self.split=split

    def dataset_split(self, images_dir):
        images_dir=images_dir
        full_dataset=os.listdir(images_dir)
        train_size = int(self.train_factor * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset=torch.utils.data.random_split(full_dataset, [train_size, valid_size])

        # split the dataset to 'train' and 'valid' set
        train_dir=os.path.join(self.root, 'images/train_{}'.format(self.train_factor))
        valid_dir = os.path.join(self.root, 'images/valid_{}'.format(round(1-self.train_factor, 2)))
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
        for fname in train_dataset:
            img_src=os.path.join(images_dir, fname)
            img_dst=os.path.join(train_dir, fname)
            shutil.copyfile(img_src, img_dst)
        for fname in valid_dataset:
            img_src = os.path.join(images_dir, fname)
            img_dst = os.path.join(valid_dir, fname)
            shutil.copyfile(img_src, img_dst)
        print('=> split dataset finished!')
        return train_dataset, valid_dataset

    def images_dict(self, images_dir, dataset):  # use pytorch to split dataset, get the filenames
        imglist = dataset
        images = []
        for fname in imglist:
            img = cv2.imread(os.path.join(images_dir, fname))
            height = img.shape[0]
            width = img.shape[1]
            img_id = fname.split('.')[0].lstrip('0')
            img_id = img_id if img_id != '' else '0'
            image = {
                "license": 1,
                "id": int(img_id),
                "width": int(width),
                "height": int(height),
                "file_name": fname,
                "flickr_url": '',
                "coco_url": '',
                "data_captured": ''
            }
            images.append(image)
        return images

    def annotations_dict(self, dataset, json_path):
        imglist = dataset
        json_file = open(json_path, encoding='utf-8').read()
        self.anno = json.loads(json_file)

        annotations=[]
        n=0
        for fname in imglist:
            img_id=fname.split('.')[0].lstrip('0')
            img_id = img_id if img_id != '' else '0'
            anno_id = self._get_anno_index(fname)

            assert anno_id != -1, 'There is no image annotation!'
            persons = self.anno[anno_id]['persons']
            for i in range(len(persons)):
                track_id=persons[i]['track_id'][0]
                bbox=persons[i]['bbox']
                score=bbox[4]
                bbox=[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

                keypoints=persons[i]['keypoints']
                keypoints.sort(key=lambda x: int(x[0]))   # joint order
                keypoints=np.array(keypoints)[:, 1:4]
                num_keypoints=sum([1 if kpt[-1]>0 else 0 for kpt in keypoints])  # visible joints number
                coco_keypoints=[]
                for kpt in keypoints:
                    coco_keypoints += list(kpt)   # [x,y,v,x,y,v,...]

                coco_keypoints=[int(x) for x in coco_keypoints]

                annotation={
                    "id": int(n),     # There may be multiple people in one image
                    "track_id": int(track_id),
                    "image_id": int(img_id),
                    "category_id": 1,
                    "segmentation": [[0, 0, 0, 0, 0, 0, 0, 0]],
                    "area": round(bbox[2]*bbox[3], 2),  # keep two decimal places
                    "bbox": list(bbox),
                    "score": float(score),
                    "iscrowd": 1,
                    "keypoints": list(coco_keypoints),
                    "num_keypoints": int(num_keypoints)
                }
                n += 1
                annotations.append(annotation)
        return annotations

    def _get_anno_index(self, file_name):
        for i in range(len(self.anno)):
            if self.anno[i]['file_name'] == file_name:
                return i
        else:
            return -1

    def create_json(self):
        images_dir = os.path.join(self.root, 'HIE20/videos/train/images/HIE20')
        json_path=os.path.join(self.root, 'HIE20/labels/train/track2&3/HIE20.json')

        if self.split:
            train_dataset, valid_dataset = self.dataset_split(images_dir)
        else:
            train_dataset=os.listdir(os.path.join(self.root,  'images/train_{}'.format(self.train_factor)))
            valid_dataset=os.listdir(os.path.join(self.root, 'images/valid_{}'.format(round(1-self.train_factor, 2))))
            print('train_dataset length:', len(train_dataset))
            print('valid_dataset length:', len(valid_dataset))

        train_images=self.images_dict(images_dir, train_dataset)
        train_annotations=self.annotations_dict(train_dataset, json_path)
        train_path=os.path.join(self.root, 'annotations/train_{}.json'.format(self.train_factor))

        valid_images = self.images_dict(images_dir, valid_dataset)
        valid_annotations = self.annotations_dict(valid_dataset, json_path)
        valid_path = os.path.join(self.root, 'annotations/valid_{}.json'.format(round(1-self.train_factor, 2)))

        train_results={
            "info": {"year": 2020,
                     "version": '',
                     "description": '',
                     "contributor": '',
                     "url": '',
                     "data_created": ''
                     },
            "licenses": [{
                "id": 1,
                "name": '',
                "url": ''
            }],
            "images": train_images,
            "annotations": train_annotations,
            "categories": [{
                "id": 1,
                "name": 'person',
                "supercategory": "person",
                "keypoints": ['nose', 'chest', 'right_shoulder', 'right_elbow', 'right_wrist',
                            'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
                            'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'],
                "skeleton": [['nose', 'chest'], ['chest', 'right_shoulder'], ['right_shoulder', 'right_elbow'],
                    ['right_elbow', 'right_wrist'], ['chest', 'left_shoulder'], ['left_shoulder', 'left_elbow'],
                    ['left_elbow', 'left_wrist'], ['chest', 'right_hip'], ['right_hip', 'right_knee'],
                    ['right_knee', 'right_ankle'], ['chest', 'left_hip'], ['left_hip', 'left_knee'],
                    ['left_knee', 'left_ankle']]
            }]
        }

        valid_results = {
            "info": {"year": 2020,
                     "version": '',
                     "description": '',
                     "contributor": '',
                     "url": '',
                     "data_created": ''
                     },
            "images": valid_images,
            "annotations": valid_annotations,
            "licenses": [{
                "id": 1,
                "name": '',
                "url": ''
            }],
            "categories": [{
                "id": 1,
                "name": 'person',
                "supercategory": "person",
                "keypoints": ['nose', 'chest', 'right_shoulder', 'right_elbow', 'right_wrist',
                              'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip',
                              'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle'],
                "skeleton": [['nose', 'chest'], ['chest', 'right_shoulder'], ['right_shoulder', 'right_elbow'],
                             ['right_elbow', 'right_wrist'], ['chest', 'left_shoulder'],
                             ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
                              ['chest', 'right_hip'], ['right_hip', 'right_knee'],
                             ['right_knee', 'right_ankle'], ['chest', 'left_hip'],
                             ['left_hip', 'left_knee'], ['left_knee', 'left_ankle']]
            }]
        }

        with open(train_path, 'w') as tf:
            json.dump(train_results, tf)
        print('=> create train json finished!')
        tf.close()

        with open(valid_path, 'w') as vf:
            json.dump(valid_results, vf)
        print('=> create valid json finished!')
        vf.close()



if __name__=='__main__':
    root='/media/yxq/HardDisk/datasets/HIE20'

    creator=CreateCoco(root, train_factor=0.99, split=False)
    creator.create_json()