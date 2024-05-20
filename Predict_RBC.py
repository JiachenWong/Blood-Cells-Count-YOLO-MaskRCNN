#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:51:32 2024

@author: wangjiachen
"""
import os
import  cv2
img_path = '/Users/wangjiachen/Downloads/Mask_RCNN-master/images2/val/RBC30.jpg'
image = cv2.imread( img_path )

from config import Config
import model as modellib

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

config = BalloonConfig()
model = modellib.MaskRCNN(mode='inference',config=config,model_dir='logs')
model.load_weights('/Users/wangjiachen/Downloads/Mask_RCNN-master/logs/balloon20240515T1618/mask_rcnn_balloon_0096.h5',by_name=True)

result = model.detect([image])
print(result[0])

class_names = ['BG','RBC']

from visualize import display_instances


save_path = os.path.dirname(img_path) + os.path.basename(img_path) +'实例分割.png'


display_instances(image, result[0]['rois'], result[0]['masks'], result[0]['class_ids'],
                  class_names,
                  scores=None, title="RBCs",
                  figsize=(5,5), ax=None,
                  show_mask=True, show_bbox=True,
                  colors=None, captions=None ,save_path=save_path )




