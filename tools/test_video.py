# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('../')  # test_video.py in tools/

import os
import cv2
import argparse
import torch
import torchvision
import json
import numpy as np
from collections import defaultdict

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from core.group import HeatmapParser
from utils.transforms import get_multi_scale_size
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.vis import draw_image
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results

from dataset.VideoProcess import VideoLoader, DataWriter
from dataset.HIECOCODataset import HIECOCODataset


def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('--video_path',
                        help='video path',
                        default='/media/yxq/HardDisk/datasets/HIE20/test/clip.mp4',
                        type=str)

    parser.add_argument('--data',
                        help='dataset',
                        default='HIE',
                        type=str)

    parser.add_argument('--outdir',
                        help='output directory',
                        default='../output/test',
                        type=str)

    parser.add_argument('--save_video',
                        help='save video or not',
                        default=False)

    parser.add_argument('--video_format',
                        help='save video with avi or mp4 format',
                        default='avi',
                        type=str)

    parser.add_argument('--save_img',
                        help='save vis image',
                        default=True)

    parser.add_argument('--vis',
                        help='visualize image',
                        default=False,
                        action='store_true')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def processKeypoints(keypoints):
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


def keypoints_results(keypoints):
    num_joints = 14    # *****
    annorect = []  # element: a person

    # iterate one image in a image_batch, only one cycle for batch_size=1
    for img_kpts in keypoints:  # one cycle: process all persons in a image.
        if len(img_kpts) == 0:
            continue

        _key_points = np.array(
            [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
        )
        key_points = np.zeros(
            (_key_points.shape[0], num_joints * 3),
            dtype=np.float
        )
        for ipt in range(num_joints):  # key_points, all persons in a image
            key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
            key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
            key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

        for k in range(len(img_kpts)):  # iterate a person in all persons
            # print(key_points[k])
            point = []
            for j in range(num_joints):  # one person joints
                kp = {
                    'score': [round(float(key_points[k][2 + j * 3]), 1)],
                    'x': [int(key_points[k][0 + j * 3])],
                    'y': [int(key_points[k][1 + j * 3])],
                    'id': [int(j)]
                }
                point.append(kp)

            kpt = key_points[k].reshape((num_joints, 3))
            left_top = np.amin(kpt, axis=0)
            right_bottom = np.amax(kpt, axis=0)

            # create 'annorect', one person for a 'annorect' element
            annorect.append({
                'score': [round(float(img_kpts[k]['score']), 1)],
                'x1': [int(left_top[0])],
                'y1': [int(left_top[1])],
                'x2': [int(right_bottom[0])],
                'y2': [int(right_bottom[1])],
                'annopoints': [{'point': point}]
            })
    return annorect  # all persons in a image


def person_result(preds, scores, img_id):
    file_name = '{}.jpg'.format(str(img_id).zfill(6))

    # preds is a list of: image x person x (keypoints)
    # keypoints: num_joints * 4 (x, y, score, tag)
    kpts = defaultdict(list)
    # iterate image_batch, only one image when batch_size=1: len(preds)==1
    for idx, _kpts in enumerate(preds):  # len(preds[0]) == num_person
        # iterate persons in a image, len(kpt)==num_joints==14
        for idx_kpt, kpt in enumerate(_kpts):  # len(_kpts) == num_person
            area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
            kpt = processKeypoints(kpt)
            # if self.with_center:
            if cfg.DATASET.WITH_CENTER and not cfg.TEST.IGNORE_CENTER:
                kpt = kpt[:-1]
            kpts[int(file_name[-10:-4])].append(    # hie: [-10:-4], coco: [-16:-4]
                {
                    'keypoints': kpt[:, 0:3],
                    'score': scores[idx_kpt],
                    'tags': kpt[:, 3],
                    'image': file_name,
                    'area': area
                }
            )

    # rescoring and oks nms
    oks_nmsed_kpts = []
    # image x person x (keypoints)
    for img in kpts.keys():
        # person x (keypoints)
        img_kpts = kpts[img]
        # person x (keypoints)
        # do not use nms, keep all detections
        keep = []
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    annorect = keypoints_results(oks_nmsed_kpts)
    return annorect


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'test'
    )

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, 'model_best.pth.tar')
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    HMparser = HeatmapParser(cfg)  # ans, scores

    res_folder = os.path.join(args.outdir, 'results')
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)
    video_name = args.video_path.split('/')[-1].split('.')[0]
    res_file = os.path.join(res_folder, '{}.json'.format(video_name))

    # read frames in video
    stream = cv2.VideoCapture(args.video_path)
    assert stream.isOpened(), 'Cannot capture source'


    # fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    video_dir = os.path.join(args.outdir, 'video', args.data)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    image_dir=os.path.join(args.outdir, 'images', args.data)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if args.video_format == 'mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path = os.path.join(video_dir, '{}.mp4'.format(video_name))
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(video_dir, '{}.avi'.format(video_name))

    if args.save_video:
        out = cv2.VideoWriter(video_path, fourcc, fps, frameSize)

    num = 0
    annolist = []
    while (True):
        ret, image = stream.read()
        print("num:", num)

        if ret is False:
            break

        all_preds = []
        all_scores = []

        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = HMparser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(  # joints for all persons in a image
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        image=draw_image(image, final_results, dataset=args.data)
        all_preds.append(final_results)
        all_scores.append(scores)

        img_id = num
        num += 1
        file_name = '{}.jpg'.format(str(img_id).zfill(6))
        annorect = person_result(all_preds, scores, img_id)

        annolist.append({
            'annorect': annorect,
            'ignore_regions': [],
            'image': [{'name': file_name}]
        })
        # print(annorect)

        if args.save_video:
            out.write(image)

        if args.save_img:
            img_path = os.path.join(image_dir, file_name)
            cv2.imwrite(img_path, image)

    final_results = {'annolist': annolist}
    with open(res_file, 'w') as f:
        json.dump(final_results, f)
    print('=> create test json finished!')

    # print('=> finished! you can check the output video on {}'.format(save_path))
    stream.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
