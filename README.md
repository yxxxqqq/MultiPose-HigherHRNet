# MultiPose-HigherHRNet
##### Description

- Realized crowd multi-person pose estimation based on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). 

- Mainly deal with the dataset on [ACM MM Grand Challenge on Large-scale Human-centric Video Analysis in Complex Events](http://humaninevents.org/) 

##### Installation

- Install dependencies:

  ```
  pip install -r requirements.txt
  ```

- Install [COCOAPI](https://github.com/cocodataset/cocoapi) and [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) 

- My local environment

  ```
  local host: torch1.4-py36
  71 server: docker run -it --runtime=nvidia --shm-size=64G -v /home/yxq:/home/yxq yxq/multipose:cuda10 /bin/bash
  ```

- Your directory tree should look like this:

  ```
  ${POSE_ROOT}
  ├── data
  ├── experiments
  ├── lib
  ├── log
  ├── models
  ├── output
  ├── tools 
  ├── README.md
  └── requirements.txt
  ```

##### Data preparation

- **For HIE20 data**, the dataset has its .json format, I write my dataloader.py, but for evaluation, the original code use COCO format, so I rewrite the HIE .json format same as COCO. 

##### Training and Testing

- For single-scale testing:

  ```
  python tools/valid.py --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth
  ```

- Training on COCO train2017 dataset

  ```
  python tools/dist_train.py \
      --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
  ```

- Training on HIE20 dataset

  ```
  python tools/dist_train.py --cfg experiments/hie/higher_hrnet/w32_512_adam_lr1e-3.yaml 
  ```

- Valid on HIE20 dataset

  ```
  python tools/valid.py --cfg experiments/hie/higher_hrnet/w32_512_adam_lr1e-3.yaml  TEST.MODEL_FILE models/pytorch/pose_coco/pose_higher_hrnet_w32_512.pth  TEST.FLIP_TEST False
  ```

- Test on HIE20 dataset (has no groundtruth), and find the results in 'output'

  ```
  cd tools
  python test_video.py --cfg ../experiments/hie/higher_hrnet/w32_512_adam_lr1e-3.yaml --video_path /media/yxq/HardDisk/datasets/HIE20/test/26.mp4
  ```

- Tensorboard

  ```
  tensorboard --logdir=log/hie_kpt/pose_higher_hrnet
  ```

 ##### My experiments

- Run xxx.py in pycharm

  ```
  add '../' path in train.py
  ```

- Set CUDA

  ```
  import os
  os.environ['CUDA_VISIBLE_DEVICES']='1,2,3' 
  ```

  

