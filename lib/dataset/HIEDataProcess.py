#-*-coding:utf-8-*-
import os
import cv2
import argparse
import json
import shutil
import numpy as np

import torch

# step 1: video to image
def video_to_image(args):
    if not os.path.exists(args.image_dir):
        os.mkdir(args.image_dir)

    cap=cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError('can not open video')

    n=0
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        image_path='{}/{}.jpg'.format(args.image_dir, str(n).zfill(6))  # 000000.jpg
        cv2.imwrite(image_path, frame)
        n+=1
    print('=> images saved!')
    cap.release()


# step 2: rename all images and save to merge directory
def merge_images():
    root_dir='/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images'
    merge_dir='/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images/merge'
    img_dirs=os.listdir(root_dir)
    img_dirs.sort(key=lambda x: int(x))

    if not os.path.exists(merge_dir):
        os.mkdir(merge_dir)

    num=0
    for i in range(len(img_dirs)):
        img_dir=os.path.join(root_dir, img_dirs[i])
        img_list=os.listdir(img_dir)
        img_list.sort(key=lambda x: int(x.split('.')[0]))
        for j in range(len(img_list)):
            img_src=os.path.join(img_dir, img_list[j])
            img_dst = os.path.join(merge_dir, img_list[j])
            img_rename = os.path.join(merge_dir, '{}.jpg'.format(str(num).zfill(12)))

            shutil.copyfile(img_src, img_dst)
            os.rename(img_dst, img_rename)
            num += 1
        print("=> {} image directory finish!".format(i))


# step 3: parser original json and rewrite to merge.json
def recreate_json(json_path, start_num):
    '''
    {
    "file_name": '',
    "persons":[
        {
        "track_id": 0,
        "bbox": bbox,  # [x1, y1, x2, y2, score]
        "keypoints": joints-14  # [ [index, x, y, score], [], [] ... ]
        },
    ]
    }
    '''
    json_file = open(json_path, encoding='utf-8').read()
    json_dict = json.loads(json_file)
    joint_idx = list(range(0, 14))   # index: 0~13

    # for all images in a json file
    images=[]
    for img in json_dict['annolist']:
        file_name=img['image'][0]['name']

        # for all persons in a image
        persons = []
        for obj in img['annorect']:
            bbox=[obj['x1'][0], obj['y1'][0], obj['x2'][0], obj['y2'][0], obj['score'][0]]   # x1, y1, x2, y2, score
            track_id=obj['track_id']

            # for all joints in a person
            joints = []
            points=obj['annopoints'][0]['point']
            for j in range(len(points)):
                # print(points[j])
                joint = [points[j]['id'][0], points[j]['x'][0],
                         points[j]['y'][0], points[j]['score'][0]]
                joints.append(joint)

            if len(joints) < 14:
                exist = []
                for j in range(len(joints)):
                    exist += [int(joints[j][0])]
                unexist= list(set(joint_idx).difference(set(exist)))
                for idx in unexist:
                    joints.insert(idx, [idx, 0, 0, 0])

            # for one person
            person={
                "track_id": track_id,
                "bbox": bbox,  # [x1, y1, x2, y2, score]
                "keypoints": joints  # [ [index, x, y, score], [], [] ... ]
            }
            persons.append(person)

        file_name=int(file_name.split('.')[0]) + start_num
        file_name='{}.jpg'.format(str(file_name).zfill(12))
        image={
            "file_name": file_name,
            "persons": persons
        }
        images.append(image)
    return images

def merge_jsons():
    root_dir='/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3'
    save_path='/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3/merge.json'
    json_paths=os.listdir(root_dir)
    json_paths.sort(key=lambda x: int(x.split('.')[0]))

    final_json=[]
    num=0
    for i in range(len(json_paths)):
        json_path=os.path.join(root_dir, json_paths[i])
        images=recreate_json(json_path, num)
        final_json += images
        num += len(images)
        print('=> merge {} json finished!'.format(i))
    with open(save_path, 'w') as f:
        json.dump(final_json, f)
    print('=> rewrite json finished!')


# step 4: visualization
def vis_merge_result():
    image_path='/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images/HIE20/000000002797.jpg'
    json_path='/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3/HIE20.json'
    save_dir='/media/yxq/HardDisk/datasets/HIE20/HIE20/merge_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = cv2.imread(image_path)
    json_file = open(json_path, encoding='utf-8').read()
    images = json.loads(json_file)
    file_name=image_path.split('/')[-1]

    for i in range(len(images)):
        if images[i]['file_name']==file_name:
            persons = images[i]['persons']

            for j in range(len(persons)):
                points=persons[j]['keypoints']
                bbox=persons[j]['bbox']
                print(points)
                print(bbox)
                joints=[]
                color = np.random.randint(0, 255, size=3)
                color = tuple([int(x) for x in color])
                for joint in points:
                    joints.append((int(joint[1]), int(joint[2]), joint[3]))

                cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[1]), color, 1)
                cv2.line(image, (bbox[2], bbox[1]), (bbox[2], bbox[3]), color, 1)
                cv2.line(image, (bbox[2], bbox[3]), (bbox[0], bbox[3]), color, 1)
                cv2.line(image, (bbox[0], bbox[3]), (bbox[0], bbox[1]), color, 1)

                for pt in joints:
                    cv2.circle(image, (pt[0], pt[1]), 1, color, 2)
                    # cv2.putText(image, str(k), pt, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    # cv2.putText(image, str(pt[2]), (pt[0], pt[1]), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    cv2.imshow('img', image)
    k=cv2.waitKey(5000)
    if k == ord('s'):
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, image)
        cv2.destroyAllWindows()
    if k == ord('t'):
        cv2.imshow('img', image)
        cv2.waitKey(0)


def vis_gt_result():
    image_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images/1/001491.jpg'
    json_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3/1.json'
    save_dir = '/media/yxq/HardDisk/datasets/HIE20/HIE20/gt_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = cv2.imread(image_path)
    json_file = open(json_path, encoding='utf-8').read()
    json_dict = json.loads(json_file)
    file_name = image_path.split('/')[-1]

    for img in json_dict['annolist']:
        if img['image'][0]['name']==file_name:
            for obj in img['annorect']:
                bbox = [int(obj['x1'][0]), int(obj['y1'][0]), int(obj['x2'][0]), int(obj['y2'][0]), obj['score'][0]]
                points = obj['annopoints'][0]['point']
                joints = []
                print(points)
                print(bbox)
                color = np.random.randint(0, 255, size=3)
                color = tuple([int(x) for x in color])
                for j in range(len(points)):
                    # print(points[j])
                    joint = [points[j]['id'][0], int(points[j]['x'][0]),
                             int(points[j]['y'][0]), points[j]['score'][0]]
                    joints.append(joint)

                cv2.line(image, (bbox[0], bbox[1]), (bbox[2], bbox[1]), color, 1)
                cv2.line(image, (bbox[2], bbox[1]), (bbox[2], bbox[3]), color, 1)
                cv2.line(image, (bbox[2], bbox[3]), (bbox[0], bbox[3]), color, 1)
                cv2.line(image, (bbox[0], bbox[3]), (bbox[0], bbox[1]), color, 1)

                for i in range(len(joints)):
                    # print(joints[i])
                    cv2.circle(image, (joints[i][1], joints[i][2]), 1, color, 2)
                    # cv2.putText(image, str(i), (joints[i][1], joints[i][2]), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                    # cv2.putText(image, str(joints[i][3]), (joints[i][1], joints[i][2]), cv2.FONT_HERSHEY_PLAIN, 1,
                    #             color, 1)
                    i += 1

    cv2.imshow('img', image)
    k = cv2.waitKey(5000)
    if k == ord('s'):
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, image)
        cv2.destroyAllWindows()
    if k == ord('t'):
        cv2.imshow('img', image)
        cv2.waitKey(0)



def dataset_split(full_dataset):
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    return train_dataset, test_dataset






if __name__=='__main__':
    parser=argparse.ArgumentParser(description='HIE dataset process')
    parser.add_argument('--video_path', default='/media/yxq/HardDisk/datasets/HIE20/test/20.mp4',
                        type=str, metavar='PATH', help='path to HIE video')
    parser.add_argument('--image_dir', default='/media/yxq/HardDisk/datasets/HIE20/test/20',
                        type=str, metavar='PATH', help='directory to save images')

    '''step 1: video to images '''
    # video_to_image(parser.parse_args())

    '''step 2: rename all images and save to merge directory '''
    # merge_images()

    '''step 3: parser original json and rewrite to merge.json'''
    # merge_jsons()

    '''step 4: visualization '''
    # vis_merge_result()
    # vis_gt_result()



