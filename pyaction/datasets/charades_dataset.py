#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import numpy as np
import torch

from . import charades_helper as charades_helper
from . import cv2_transform as cv2_transform
from . import transform as transform
from . import utils as utils
from .build import DATASET_REGISTRY

import random
import os

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Charades(torch.utils.data.Dataset):
    """
    Charades Dataset
    """

    def __init__(self, cfg, split):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = cfg.MODEL.NUM_CLASSES
        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.CHARADES.BGR
        if self._split == "train":
            self._crop_size = cfg.DATA.TRAIN_CROP_SIZE
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.CHARADES.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.CHARADES.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.CHARADES.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.CHARADES.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = cfg.DATA.TEST_CROP_SIZE
            self._test_force_flip = cfg.CHARADES.TEST_FORCE_FLIP

        self._load_data(cfg)

    def _load_data(self, cfg):
        """
        Load frame paths and annotations from files

        Args:
            cfg (CfgNode): config
        """
        # Loading frame paths.
        # (self._image_paths, self._video_idx_to_name,) = charades_helper.load_image_lists(
        #     cfg, is_train=(self._split == "train")
        # )
        #Qiuyue
        list_filenames = [
            os.path.join(cfg.CHARADES.FRAME_LIST_DIR, filename)
            for filename in (
                cfg.CHARADES.TRAIN_LISTS
                if self._split == 'train'
                else cfg.CHARADES.TEST_LISTS)
        ]
        (self._image_paths,
         self._image_labels,
         self._video_idx_to_name, _) = charades_helper.load_image_lists(
            list_filenames) 
        
        if self._split != 'train':
            # Charades is a video-level task.
            self._convert_to_video_level_labels()

        self._num_videos = len(self._image_paths)
        #Qiuyue

        self.print_summary()

    def _convert_to_video_level_labels(self):
        for video_id in range(len(self._image_labels)):
            video_level_labels = aggregate_labels(self._image_labels[video_id])
            for i in range(len(self._image_labels[video_id])):
                self._image_labels[video_id][i] = video_level_labels

    def print_summary(self):
        logger.info("=== CHARADES dataset summary ===")
        logger.info("Split: {}".format(self._split))
        logger.info("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        logger.info("Number of frames: {}".format(total_frames))
        logger.info("Number of key frames: {}".format(len(self)))
        logger.info("Number of boxes: {}.".format(self._num_boxes_used))

    def __len__(self):
        # return len(self._keyframe_indices)
        if self._split == 'train':
            return len(self._image_paths)
        else:
            return len(self._image_paths) * self.cfg.CHARADES.NUM_TEST_CLIPS #TODO

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip with opencv as backend.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """

        height, width, _ = imgs[0].shape

        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        # `transform.py` is list of np.array. However, for AVA, we only have
        # one np.array.
        boxes = [boxes]

        # The image now is in HWC, BGR format.
        if self._split == "train":  # "train"
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            # random flip
            imgs, boxes = cv2_transform.horizontal_flip_list(
                0.5, imgs, order="HWC", boxes=boxes
            )
        elif self._split == "val":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            # Short side to test_scale. Non-local and STRG uses 256.
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(self._crop_size, boxes[0], height, width)
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError("Unsupported split mode {}".format(self._split))

        # Convert image to CHW keeping BGR order.
        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        # Image [0, 255] -> [0, 1].
        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                # img.reshape((3, self._crop_size, self._crop_size))
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        # Concat list of images to single ndarray.
        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2],
        )
        return imgs, boxes

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes = transform.clip_boxes_to_image(boxes, height, width)

        if self._split == "train":
            # Train split
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = transform.random_crop(imgs, self._crop_size, boxes=boxes)

            # Random flip.
            imgs, boxes = transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._split == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._split == "test":
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = transform.random_short_side_scale_jitter(
                imgs, min_size=self._crop_size, max_size=self._crop_size, boxes=boxes,
            )

            if self._test_force_flip:
                imgs, boxes = transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError("{} split not supported yet!".format(self._split))

        # Do color augmentation (after divided by 255.0).
        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = transform.color_jitter(
                    imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4,
                )

            imgs = transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        boxes = transform.clip_boxes_to_image(boxes, self._crop_size, self._crop_size)

        return imgs, boxes


    def get_minibatch_info(self, indices): #Qiuyue
        """
        Given iteration indices, return the necessarry information for
        constructing a minibatch. This will later be used in charades_data_input.py
        to actually load the data and constructing blobs.
        """
        half_len = self._seq_len // 2

        image_paths = []
        labels = []
        spatial_shift_positions = []

        if not isinstance(indices, list):
            indices = indices.tolist()

        while len(indices) < self._batch_size // self.cfg.NUM_GPUS: #TODO
            indices.append(indices[0])

        for idx in indices:

            # center_idx is the middle frame in a clip.
            video_idx = idx % self._num_videos
            num_frames = len(self._image_paths[video_idx])
            if self._split == 'train':
                center_idx = sample_train_idx(num_frames, self._seq_len)
                spatial_shift_positions.append(None)
            else:
                # for, e.g., 30-clip testing, multi_clip_idx stands for
                # (0-left, 0-center, 0-right, ... 9-left, 9-center, 9-right)
                multi_clip_idx = idx // self._num_videos

                spatial_shift_positions.append(multi_clip_idx % 3)
                segment_id = multi_clip_idx // 3

                center_idx = sample_center_of_segments(
                    segment_id, num_frames, self._num_test_segments, half_len)

            seq = utils.get_sequence(
                center_idx, half_len, self._sample_rate, num_frames)

            image_paths.append([self._image_paths[video_idx][frame]
                                for frame in seq])
            labels.append(aggregate_labels(
                [self._image_labels[video_idx][frame]
                 for frame in range(seq[0], seq[-1] + 1)]))

        split_list = [self._split_num] * len(indices)

        return (image_paths, labels, split_list, spatial_shift_positions)

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        """
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        # Get the frame idxs for current clip.
        seq = utils.get_sequence(
            center_idx,
            self._seq_len // 2,
            self._sample_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        
        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]
        assert len(clip_label_list) > 0

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            boxes.append(box_labels[0])
            labels.append(box_labels[1])
        boxes = np.array(boxes)
        # Score is not used.
        boxes = boxes[:, :4].copy()
        ori_boxes = boxes.copy()

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        """
        indices = idx#TODO
        boxes = None
        # image_paths, labels, split_list, spatial_shift_positions = self.get_minibatch_info(indices) 
        image_paths, labels, split_list, _ = self.get_minibatch_info(indices) 
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.CHARADES.IMG_PROC_BACKEND
        )
        if self.cfg.CHARADES.IMG_PROC_BACKEND == "pytorch":
            # T H W C -> T C H W.
            imgs = imgs.permute(0, 3, 1, 2)
            # Preprocess images and boxes.
            imgs, boxes = self._images_and_boxes_preprocessing(imgs, boxes=boxes)
            # imgs, _ = data_input_helper.images_and_boxes_preprocessing(
            #     imgs, split, crop_size, spatial_shift_pos)
            # T C H W -> C T H W.
            imgs = imgs.permute(1, 0, 2, 3)
        else:
            # Preprocess images and boxes
            imgs, boxes = self._images_and_boxes_preprocessing_cv2(imgs, boxes=boxes)

        # Construct label arrays.
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1

        imgs = utils.pack_pathway_output(self.cfg, imgs)
        # metadata = [[video_idx, sec]] * len(boxes)

        # extra_data = {
        #     "boxes": boxes,
        #     # "ori_boxes": ori_boxes,
        #     "metadata": metadata,
        # }

        return imgs, label_arrs, idx#, extra_data


def sample_train_idx(num_frames, seq_len):
    """Sample training frames."""
    half_len = seq_len // 2
    if num_frames < seq_len:
        center_idx = num_frames // 2
    else:
        center_idx = random.randint(
            half_len, num_frames - half_len)
    return center_idx


def sample_center_of_segments(segment_id, num_frames,
                              num_test_segments, half_len):
    """Sample testing clips to be the center of uniformly split segments."""
    center_idx = int(np.round(
        (float(num_frames) / num_test_segments)
        * (segment_id + 0.5)))

    return center_idx


def aggregate_labels(label_list):
    """Aggregate a sequence of labels."""
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))
