#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import numpy as np
import torch
import torch.utils.data

import pyaction.utils.logging as logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Minikinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """

        # square video resize
        self.square_jitter = cfg.META.DATA.SQUARE_JITTER
        # Unified test metric
        self.unified_eval = cfg.TEST.UNIFIED_EVAL
        # Center-crop & multi-view
        self.center_crop_multi_view = cfg.TEST.CENTER_CROP_MULTI_VIEW

        # Conflit
        assert not (self.unified_eval and self.center_crop_multi_view)

        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg
        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            if self.unified_eval:
                self._num_clips = 1
            elif self.center_crop_multi_view:
                self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * 1
            else:
                self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        # few-shot
        self.classes_per_set = cfg.META.SETTINGS.N_SUPPORT_WAY
        self.samples_per_class = cfg.META.SETTINGS.K_SUPPORT_SHOT

        self.use_replaced_cmn_list = cfg.DATA.USE_REPLACED_CMN

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # use test split for validation during training for convienience.
        if self.mode == 'val' and self.cfg.META.DATA.TEST_AS_VAL:
            split_name = 'test'
        else:
            split_name = self.mode
        
        split_filename = "{}{}.csv".format(split_name, self.cfg.META.DATA.CSV_SUFFIX)
        logger.info(
            "Current Mode:{}, used split: {}".format(
                self.mode, split_filename
            )
        )
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            split_filename
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(path_to_file)

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2
                path, label = path_label.split()
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file  ##### where's self._split_idx's definition?
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

        # For few-shot learning
        self.data_classes = []  # list of data lists for each class
        for _ in range(max(self._labels)+1):
            self.data_classes.append([])
        for i in range(len(self._labels)):
            path = self._path_to_videos[i]
            label = self._labels[i]
            spatial_temporal_idx = self._spatial_temporal_idx[i]
            self.data_classes[label].append({"idx": i, "path": path, "spatial_temporal_idx": spatial_temporal_idx})
        self.n_classes = len(self.data_classes)

    def __getitem__(self, index):
        """
        Given the task index(useless?), return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # select a non-overlapping label set
        classes = np.random.choice(self.n_classes, self.classes_per_set, False)
        # select classes for each test sample
        x_hat_class = np.random.choice(classes, self.samples_per_class, True)

        support_x = []
        support_y = []
        target_x = []
        target_y = []

        # For sanity check
        support_y_real = []
        target_y_real = []

        for i, cur_class in enumerate(classes):  # each class
            # Count number of times this class is inside the meta-test
            n_test_samples = np.sum(cur_class == x_hat_class)
            # example_idxs = np.random.choice(len(self.data_classes[cur_class]), self.samples_per_class + n_test_samples + self._num_retries, False)

            example_idxs = np.random.permutation(len(self.data_classes[cur_class]))

            # Cursor for popping example_idxs
            j = 0

            # Construct support
            for _ in range(self.samples_per_class):
                for __ in range(self._num_retries):
                    if j >= len(example_idxs):
                        raise RuntimeError("Running out of candidate videos.")
                    eid = example_idxs[j]
                    absolute_eid = self.data_classes[cur_class][eid]["idx"]
                    # print(cur_class, eid, absolute_eid, "\n")
                    example_video = self._get_video(absolute_eid)  # dict
                    j += 1
                    if example_video:
                        break
                if not example_video:
                    raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))

                # absolute label
                support_y_real.append(example_video["label"])

                # relative label
                example_video["label"] = i
                support_x.append(example_video["frames"][0])
                support_y.append(example_video["label"])

            # Construct query
            for _ in range(n_test_samples):
                for __ in range(self._num_retries):
                    if j >= len(example_idxs):
                        raise RuntimeError("Running out of candidate videos.")
                    eid = example_idxs[j]
                    absolute_eid = self.data_classes[cur_class][eid]["idx"]
                    example_video = self._get_video(absolute_eid)  # dict
                    j += 1
                    if example_video:
                        break
                if not example_video:
                    raise RuntimeError("Failed to fetch video after {} retries.".format(self._num_retries))

                # absolute label
                target_y_real.append(example_video["label"])

                # relative label
                example_video["label"] = i
                target_x.append(example_video["frames"][0])
                target_y.append(example_video["label"])

        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        target_x = torch.stack(target_x)
        target_y = torch.tensor(target_y)

        support_y_real = torch.tensor(support_y_real)
        target_y_real = torch.tensor(target_y_real)

        return support_x, support_y, target_x, target_y

    def _get_video(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:

            if self.unified_eval:
                temporal_sample_index = 0
                spatial_sample_index = 1
            elif self.center_crop_multi_view:
                temporal_sample_index = self._spatial_temporal_idx[index]
                spatial_sample_index = 1
            else:
                temporal_sample_index = (
                    self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS
                )
                # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
                # center, or right if width is larger than height, and top, middle,
                # or bottom if height is larger than width.
                spatial_sample_index = (
                    self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                )

            # min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3

            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.

            # assert len({min_scale, max_scale, crop_size}) == 1

            min_scale = max_scale = self.cfg.DATA.TEST_SCALE_SIZE
            crop_size = self.cfg.DATA.TEST_CROP_SIZE

        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        
        video_container = None
        try:
            video_container = container.get_video_container(
                self._path_to_videos[index],
                self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
            )
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    self._path_to_videos[index], e
                )
            )
            return False

        # Decode video. Meta info is used to perform selective decoding.
        frames = decoder.decode(
            video_container,
            self.cfg.DATA.SAMPLING_RATE,
            self.cfg.DATA.NUM_FRAMES,
            temporal_sample_index,
            1 if self.unified_eval else self.cfg.TEST.NUM_ENSEMBLE_VIEWS,  # self.cfg.TEST.NUM_ENSEMBLE_VIEWS, does not affect train/val
            video_meta=self._video_meta[index],
            target_fps=30,
        )

        # If decoding failed (wrong format, video is too short, and etc),
        if frames is None:
            return False

        # Perform color normalization.
        frames = frames.float()
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD)
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
        )

        label = self._labels[index]
        frames = utils.pack_pathway_output(self.cfg, frames)
        # return frames, label, index, {}
        return {"frames": frames, "label": label, "index": index}

    
    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        # return len(self._path_to_videos)
        if self.mode in ["train"]:
            return self.cfg.META.DATA.NUM_TRAIN_TASKS
        elif self.mode in ["val"]:
            return self.cfg.META.DATA.NUM_VAL_TASKS
        elif self.mode in ["test"]:
            return self.cfg.META.DATA.NUM_TEST_TASKS


    def spatial_sampling(
        self, frames, spatial_idx=-1, min_scale=256, max_scale=320, crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]

        if spatial_idx == -1:

            if hasattr(self, "square_jitter") and self.square_jitter:
                frames = torch.nn.functional.interpolate(
                    frames, size=(min_scale, min_scale), mode="bilinear", align_corners=False,
                )
            else:
                frames, _ = transform.random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )

            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            # assert len({min_scale, max_scale, crop_size}) == 1
            
            if self.debug:
                fr_sh_before = frames.shape
            
            if hasattr(self, "square_jitter") and self.square_jitter:
                frames = torch.nn.functional.interpolate(
                    frames, size=(min_scale, min_scale), mode="bilinear", align_corners=False,
                )
            else:
                frames, _ = transform.random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )

            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        
        return frames
