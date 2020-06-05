#-*-coding:utf-8-*-
import numpy as np
import datetime
import time
from collections import defaultdict
import copy
import sys
import os
import json

class HIEeval:

    def __init__(self, root, dataset, hieGt=None, cocoDt=None, iouType='keypoints'):
        self.root=root
        self.dataset=dataset
        self.hieGt=hieGt    # ground truth for hie format
        self.cocoDt=cocoDt  # detections for coco format
        self.iouType=iouType

        self.params={}     # evaluation parameters
        # per-image per-category evaluation results [KxAxI] elements
        self.evalImgs=defaultdict(list)
        self.eval={}  # accumulated evaluation results
        self._gts=defaultdict(list)    # gt for evaluation
        self._dts=defaultdict(list)    # dt for evaluation
        self._paramesEval={}    # parameters for evaluation
        self.stats=[]    # result summarization
        self.ious={}     # ious between all gts and dts

        self.anno=self._get_anno_file_name()
        self.ids = list(range(0, len(self.dataset)))


    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        anno_path = os.path.join(self.root, 'annotations', 'person_keypoints.json')
        json_file = open(anno_path, encoding='utf-8').read()
        json_dict = json.loads(json_file)
        return json_dict

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic =time.time()
        print('Running per image evaluation...')
        if self.iouType=='bbox':
            computeIoU=self.computeIoU
        elif self.iouType=='keypoints':
            computeIoU=self.computeOks



    def computeIoU(self, imgId):
        pass


