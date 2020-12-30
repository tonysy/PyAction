#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import json
import numpy as np
import os
import random
from collections import defaultdict

import torch
import torch.utils.data
from fvcore.common.file_io import PathManager

import pyaction.utils.logging as logging

from . import utils as utils
from . import transform as transform
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Minissv2(torch.utils.data.Dataset):
    """
    Something-Something v2 (SSV2) video loader. Construct the SSV2 video loader,
    then sample clips from the videos. For training and validation, a single
    clip is randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Something-Something V2 data (frame paths, labels, etc. ) to a given
        Dataset object. The dataset could be downloaded from Something-Something
        official website (https://20bn.com/datasets/something-something).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries for reading frames from disk.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Something-Something V2".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self.square_jitter = cfg.META.DATA.SQUARE_JITTER
        self.unified_eval = cfg.META.DATA.UNIFIED_EVAL

        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

        # few-shot
        self.n_support_way = cfg.META.SETTINGS.N_SUPPORT_WAY
        self.k_support_shot = cfg.META.SETTINGS.K_SUPPORT_SHOT
        self.n_query_way = cfg.META.SETTINGS.N_QUERY_WAY
        self.k_query_shot = cfg.META.SETTINGS.K_QUERY_SHOT

        logger.info("Constructing Something-Something V2 {}...".format(mode))

        with PathManager.open(
            os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR,
                "video_id_to_label.json",
            ),
            "r",
        ) as f:
            self.video_id_to_label = json.load(f)

        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # Loading video frames.
        split_filename = "fewshot_{}{}.json".format(
            self.mode, self.cfg.META.DATA.CSV_SUFFIX
        )
        video_list_file = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, split_filename)
        logger.info(
            "Current Mode:{}, used split: {}".format(self.mode, video_list_file)
        )

        with PathManager.open(video_list_file, "r") as f:
            self.video_frames_dict = json.load(f)

        # self._sem_labels = []
        self.origin_data_classes = defaultdict(list)
        for video_id in self.video_frames_dict.keys():
            sem_label = self.video_id_to_label[video_id]
            # self._sem_labels.append(sem_label)
            self.origin_data_classes[sem_label].append(video_id)

        label_mapping = {
            sem_label: i for i, sem_label in enumerate(self.origin_data_classes.keys())
        }

        self.data_classes = dict()
        for key, value in self.origin_data_classes.items():
            self.data_classes[label_mapping[key]] = value

        self.n_classes = len(self.data_classes)

        logger.info(
            "Something-Something V2 dataloader constructed "
            " (size: {}) from {}".format(len(self.video_frames_dict), video_list_file)
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
        """
        support_classes = np.random.choice(self.n_classes, self.n_support_way, False)
        query_classes = np.random.choice(support_classes, self.n_query_way, False)

        support_data, query_data = [], []
        support_meta_label, query_meta_label = [], []
        support_sem_label, query_sem_label = [], []

        for i, cur_class in enumerate(support_classes):
            example_idxs = np.random.permutation(len(self.data_classes[cur_class]))

            # Cursor for popping example_idxs
            cursor = 0
            # Construct support
            for _ in range(self.k_support_shot):
                for __ in range(self._num_retries):
                    if cursor >= len(example_idxs):
                        raise RuntimeError("Running out of candidate videos.")

                    eid = example_idxs[cursor]
                    absolute_video_id = self.data_classes[cur_class][eid]

                    clip_frames = self._get_video(absolute_video_id)[0]
                    cursor += 1

                    if clip_frames is not None:
                        break
                if clip_frames is None:
                    raise RuntimeError(
                        "Failed to fetch video after {} retries.".format(
                            self._num_retries
                        )
                    )

                support_data.append(clip_frames)
                # absolute label
                support_sem_label.append(cur_class)
                # relative label
                support_meta_label.append(i)

            # Construct query
            if cur_class in query_classes:
                for _ in range(self.k_query_shot):
                    for __ in range(self._num_retries):
                        if cursor >= len(example_idxs):
                            raise RuntimeError("Running out of candidate videos.")

                        eid = example_idxs[cursor]
                        absolute_video_id = self.data_classes[cur_class][eid]

                        clip_frames = self._get_video(absolute_video_id)[0]
                        cursor += 1

                        if clip_frames is not None:
                            break
                    if clip_frames is None:
                        raise RuntimeError(
                            "Failed to fetch video after {} retries.".format(
                                self._num_retries
                            )
                        )

                    query_data.append(clip_frames)
                    # absolute label
                    query_sem_label.append(cur_class)
                    # relative label
                    query_meta_label.append(i)

        support_data = torch.stack(support_data)
        support_meta_label = torch.tensor(support_meta_label)
        support_sem_label = torch.tensor(support_sem_label)

        query_data = torch.stack(query_data)
        query_meta_label = torch.tensor(query_meta_label)
        query_sem_label = torch.tensor(query_sem_label)

        return (
            support_data,
            support_meta_label,
            support_sem_label,
            query_data,
            query_meta_label,
            query_sem_label,
        )

    def _get_video(self, video_id):
        # short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        # if isinstance(index, tuple):
        #     index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            # # for multi-grid methods
            # if short_cycle_idx in [0, 1]:
            #     crop_size = int(
            #         round(
            #             self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
            #             * self.cfg.MULTIGRID.DEFAULT_S
            #         )
            #     )
            # if self.cfg.MULTIGRID.DEFAULT_S > 0:
            #     # Decreasing the scale is equivalent to using a larger "span"
            #     # in a sampling grid.
            #     min_scale = int(
            #         round(
            #             float(min_scale)
            #             * crop_size
            #             / self.cfg.MULTIGRID.DEFAULT_S
            #         )
            #     )
        elif self.mode in ["test"]:
            if self.unified_eval:
                spatial_sample_index = 1
            else:
                # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
                # center, or right if width is larger than height, and top, middle,
                # or bottom if height is larger than width.
                # spatial_sample_index = (
                #     self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS
                # )
                raise NotImplementedError
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        video_frames_list = self.video_frames_dict[video_id]["frames"]
        video_length = len(video_frames_list)
        num_frames = self.cfg.DATA.NUM_FRAMES
        seg_size = float(video_length - 1) / num_frames
        seq = []

        for i in range(num_frames):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if self.mode == "train":
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)
        # import pdb; pdb.set_trace()
        frames = torch.as_tensor(
            utils.retry_load_images(
                [video_frames_list[frame] for frame in seq],
                self._num_retries,
            )
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD)

        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = self.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)

        return frames

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
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
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
            if self.square_jitter:
                frames = torch.nn.functional.interpolate(
                    frames,
                    size=(min_scale, min_scale),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                frames, _ = transform.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )

            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            if self.square_jitter:
                frames = torch.nn.functional.interpolate(
                    frames,
                    size=(min_scale, min_scale),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                frames, _ = transform.random_short_side_scale_jitter(
                    frames, min_scale, max_scale
                )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
