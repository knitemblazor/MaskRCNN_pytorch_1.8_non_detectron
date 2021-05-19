import utils
from pycocotools.coco import COCO
import os
import torch
import numpy as np
from config import Config
import random
from coco_mod import *
from model import *
import pandas as pd
import cv2


class CocoConfig(Config):
    NAME = "coco"
    IMAGES_PER_GPU = 1
    # GPU_COUNT = 8
    NUM_CLASSES = 2   # COCO has 80 classes


def load_image(image_id):
    image = cv2.imread(image_id)
    if image.shape[-1] !=3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def load_mask(image_id):
    mask = np.empty([0, 0, 0])
    class_ids = np.empty([0], np.int32)
    return mask, class_ids


def load_image_gt_mod(config, image_id, augment=False, use_mini_mask=False):
    image = load_image(image_id)
    mask, class_ids = load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = utils.resize_mask(mask, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    bbox = utils.extract_bboxes(mask)
    active_class_ids = np.zeros([config.NUM_CLASSES], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_path, config, augment=True):
        df = pd.read_csv(df_path)
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(df["file_name"].to_list())
        self.error_count = 0

        self.config = config
        self.augment = augment

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                 config.RPN_ANCHOR_RATIOS,
                                                 config.BACKBONE_SHAPES,
                                                 config.BACKBONE_STRIDES,
                                                 config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        image_id = self.image_ids[image_index]
        image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
            load_image_gt_mod(self.config, image_id, augment=self.augment,
                          use_mini_mask=self.config.USE_MINI_MASK)
        if not np.any(gt_class_ids > 0):
            return None

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        images = mold_image(image.astype(np.float32), self.config)

        # Convert
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return self.image_ids.shape[0]


dataset_dir = "/home/nitheesh/Documents/projects_3/maskrcnnpytorch/pytorch-mask-rcnn/data/anno_cell_new_set.csv"
subset = "train"
config = CocoConfig()
train_set = Dataset(dataset_dir, config, augment=True)