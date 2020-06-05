# -*-coding:utf-8-*-

import os
import cv2
import json
from collections import defaultdict

def read_json(json_path):
    json_file = open(json_path, encoding='utf-8').read()
    json_dict = json.loads(json_file)

    print(len(json_dict['annolist']))
    print(json_dict['annolist'][1000])
    ann = json_dict['annolist'][1000]
    print(ann['image'])
    print(ann['annorect'])
    for i in range(len(ann['annorect'])):
        print(ann['annorect'][i])
        print(len(ann['annorect'][i]['annopoints'][0]['point']))

    # print(ann['annorect'][0])
    # for i in range(len(json_dict['annolist'])):
    #     print(json_dict['annolist'][i])


def recreate_json(json_path, start_num):
    '''
    {
    "file_name": '',
    "psersons":[
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
    joint_idx = list(range(0, 14))  # index: 0~13

    # for all images in a json file
    images = []
    for img in json_dict['annolist']:
        file_name = img['image'][0]['name']

        # for all persons in a image
        persons = []
        for obj in img['annorect']:
            bbox = [obj['x1'][0], obj['y1'][0], obj['x2'][0], obj['y2'][0], obj['score'][0]]  # x1, y1, x2, y2, score
            track_id = obj['track_id']

            # for all joints in a person
            joints = []
            points = obj['annopoints'][0]['point']
            for j in range(len(points)):
                # print(points[j])
                joint = [points[j]['id'][0], points[j]['x'][0],
                         points[j]['y'][0], points[j]['score'][0]]
                joints.append(joint)

            if len(joints) < 14:
                exist = []
                for j in range(len(joints)):
                    exist += [int(joints[j][0])]
                unexist = list(set(joint_idx).dfference(set(exist)))
                for idx in unexist:
                    joints.insert(idx, [idx, 0, 0, 0])

            # for one person
            person = {
                "track_id": track_id,
                "bbox": bbox,  # [x1, y1, x2, y2, score]
                "keypoints": joints  # [ [index, x, y, score], [], [] ... ]
            }
            persons.append(person)

        # file_name=int(file_name.split('.')[0]) + start_num
        image = {
            "file_name": file_name,
            "persons": persons
        }
        images.append(image)

    return images


def merge_jsons():
    root_dir = '/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3'
    save_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3/merge.json'
    json_paths = os.listdir(root_dir)
    json_paths.sort(key=lambda x: int(x.split('.')[0]))

    final_json = []
    num = 0
    for i in range(len(json_paths)):
        json_path = os.path.join(root_dir, json_paths[i])
        images = recreate_json(json_path, num)
        final_json += images
        num += 1
        print('=> merge {} json finished!'.format(i))
    with open(save_path, 'w') as f:
        json.dump(final_json, f)
    print('=> rewrite json finished!')


def test_alignment():
    image_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images/6/000000.jpg'
    json_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/track2&3/6.json'

    images = recreate_json(json_path, 0)
    image = cv2.imread(image_path)

    for i in range(len(images)):
        if images[i]['file_name'] == '000000.jpg':
            persons = images[i]['persons']
            points = []
            for j in range(len(persons)):
                joints = persons[j]['keypoints']
                point = []
                for joint in joints:
                    point.append((int(joint[1]), int(joint[2])))
                points.append(point)

    for i in range(len(points)):  # the number of people
        k = 0  # flag the label of number
        for pt in points[i]:
            cv2.circle(image, pt, 2, (255, 0, 0), 2)
            cv2.putText(image, str(k), pt, cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            k += 1

    cv2.imshow('img', image)
    cv2.waitKey(0)


def coco_test():
    from pycocotools.coco import COCO
    json_path = '/media/yxq/HardDisk/datasets/HIE20/annotations/valid.json'
    image_path = '/media/yxq/HardDisk/datasets/HIE20/HIE20/videos/train/images/HIE20'

    coco = COCO(json_path)
    ids = list(coco.imgs.keys())
    # print(ids)
    index = 10
    img_id = ids[index]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    target = coco.loadAnns(ann_ids)

    print(img_id)
    print(ann_ids)
    print(len(target))
    print(len(ids))
    print(target[0]['keypoints'])

    print("*********")
    print(coco)


def create_video_clip(video_path, out_path):
    frames = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    size = (int(frames.get(cv2.CAP_PROP_FRAME_WIDTH)), int(frames.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter(out_path, fourcc, 10.0, size)

    n=0
    if frames.isOpened() == False:
        print('Error opening video stream or file!')
    while (True):
        ret, frame = frames.read()

        if ret is False:
            break
        print(n)
        n += 1


        # if n < 766:
        #     out.write(frame)
        #     n += 1
        # else:
        #     break





def resize_video(video_path, save_path, img_size):
    stream = cv2.VideoCapture(video_path)
    assert stream.isOpened(), 'Cannot capture source'

    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)

    new_size=(int(img_size[0]), int(img_size[1]))
    out=cv2.VideoWriter(save_path, fourcc, fps, new_size)

    while(True):
        ret, image = stream.read()
        if ret is False:
            break

        print('original size:', image.shape)
        resied_img=cv2.resize(image, new_size)

        print('new size:', resied_img.shape)

        out.write(resied_img)
    stream.release()
    out.release()
    cv2.destroyAllWindows()








if __name__ == '__main__':
    # json_path = '/media/yxq/HardDisk/datasets/HIE20/annotations/valid.json'
    # read_json(json_path)

    # images=recreate_json(json_path, 0)
    # json_file = open(json_path, encoding='utf-8').read()
    # json_dict = json.loads(json_file)
    # print(len(json_dict))
    # print(json_dict[0])
    # test_alignment()

    # merge_jsons()

    # path='/media/yxq/HardDisk/datasets/HIE20/HIE20/labels/train/all-track/5'
    # lists=os.listdir(path)
    # lists.sort(key=lambda x: int(x.split('.')[0]))

    # for i in range(2154):
    #     if int(lists[i].split('.')[0]) != i:
    #         print(lists[i])

    coco_test()
    #
    # video_path='/media/yxq/HardDisk/datasets/HIE20/test/29_new.mp4'
    # out_path='/media/yxq/HardDisk/datasets/HIE20/test/29_new2.mp4'
    # create_video_clip(video_path, out_path)

    # json_path='/home/yxq/Workspace/CrowdMultiPose/output/test/results/31.json'
    # json_file = open(json_path, encoding='utf-8').read()
    # json_dict = json.loads(json_file)
    #
    # for img in json_dict['annolist']:
    #     print(len(img['annorect']))
        # for obj in img['annorect']:

    # video_path='/media/yxq/HardDisk/datasets/HIE20/test/clip.mp4'
    # save_path='/media/yxq/HardDisk/datasets/HIE20/test/clip_new.mp4'
    # img_size=[640, 368]
    # resize_video(video_path, save_path, img_size)




