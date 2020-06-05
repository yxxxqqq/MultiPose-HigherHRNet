# -*-coding:utf-8-*-
import os
import sys
import cv2
import numpy as np
import time

from queue import Queue
from threading import Thread

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset


class VideoLoader():
    def __init__(self, args):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.path = args.video_path
        stream = cv2.VideoCapture(self.path)
        assert stream.isOpened(), 'Cannot capture source'
        self.fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
        self.fps = stream.get(cv2.CAP_PROP_FPS)
        self.frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    def length(self):
        return self.datalen

    def videoinfo(self):
        # indicate video info
        return (self.fourcc, self.fps, self.frameSize)



class DataWriter():
    def __init__(self, args, save_path, fps, frameSize, queueSize=1024):
        self.args=args
        self.save_path=save_path
        self.save_video=args.save_video
        self.save_img=args.save_img
        self.outdir=args.outdir
        self.vis=args.vis
        self.stopped = False
        self.final_result = []

        # initialize the queue used to store frames read from video
        self.Q = Queue(maxsize=queueSize)

        if self.args.video_format == 'mp4':
            fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')

        if self.save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream=cv2.VideoWriter(self.save_path, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writting'

        if self.save_img:
            if not os.path.exists(self.outdir + '/images'):
                os.mkdir(self.outdir + '/images')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q.get()
                orig_img = np.array(orig_img, dtype=np.uint8)

                if self.save_img or self.save_video or self.vis:
                    img = orig_img
                    if self.vis:
                        cv2.imshow("HigherHRNet Demo", img)
                        cv2.waitKey(30)
                    if self.save_img:
                        cv2.imwrite(os.path.join(self.outdir, 'vis', im_name), img)
                    if self.save_video:
                        self.stream.write(img)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()