import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
# from skimage.transform import resize
import coco
import utils
import model as modellib
import visualize
import time
import torch


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = "/home/nitheesh/Documents/projects_3/maskrcnnpytorch/pytorch-mask-rcnn/models/"
# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
COCO_MODEL_PATH = "/home/nitheesh/Documents/projects_3/maskrcnnpytorch/pytorch-mask-rcnn/models/mask_rcnn_coco_0150.pth"

# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR ="/home/nitheesh/Documents/projects_3/maskrcnnpytorch/pytorch-mask-rcnn/images"


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 0
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2
    DETECTION_MIN_CONFIDENCE = 0.5


config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH,map_location=torch.device('cpu')))
print(model)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ["bg","cell"]

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
t1 = time.time()
# Run detection
results = model.detect([image])
t2 = time.time()
# Visualize results
r = results[0]
print("===",t2-t1)
from PIL import Image
img = Image.fromarray(r['masks'][5])
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

plt.show()