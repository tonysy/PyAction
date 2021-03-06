#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
import json
from statistics import mean

from pyaction.utils.timer import Timer
import pyaction.utils.distributed as du
import pyaction.datasets.ava_helper as ava_helper
import pyaction.utils.logging as logging
import pyaction.utils.metrics as metrics
import pyaction.utils.misc as misc
from pyaction.utils.ava_eval_helper import (
    evaluate_ava,
    read_csv,
    read_exclusions,
    read_labelmap,
)

logger = logging.get_logger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    Args:
        full_groundtruth(dict): list of groundtruth.
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(",")[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class AVAMeter(object):
    """
    Measure the AVA train, val, and test stats.
    """

    def __init__(self, overall_iters, cfg, mode):
        """
        overall_iters (int): the overall number of iterations of one epoch.
        cfg (CfgNode): configs.
        mode (str): `train`, `val`, or `test` mode.
        """
        self.cfg = cfg
        self.lr = None
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.full_ava_test = cfg.AVA.FULL_TEST_ON_VAL
        self.mode = mode
        self.iter_timer = Timer()
        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []
        self.overall_iters = overall_iters
        self.excluded_keys = read_exclusions(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
        )
        self.categories, self.class_whitelist = read_labelmap(
            os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE)
        )
        gt_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE)
        self.full_groundtruth = read_csv(gt_filename, self.class_whitelist)
        self.mini_groundtruth = get_ava_mini_groundtruth(self.full_groundtruth)

        _, self.video_idx_to_name = ava_helper.load_image_lists(cfg, mode == "train")

    def log_iter_stats(self, cur_epoch, cur_iter, writer=None):
        """
        Log the stats.
        Args:
            cur_epoch (int): the current epoch.
            cur_iter (int): the current iteration.
            writer (tensorboard summarywriter): writer to storage the scalars for curve.
        """

        if (cur_iter + 1) % self.cfg.LOG_PERIOD != 0:
            return

        self.iter_timer._past_durations.append(self.iter_timer.seconds())
        try:
            eta_sec = mean(self.iter_timer._past_durations[-500:])
        except Exception as e:
            print(e)
            eta_sec = mean(self.iter_timer._past_durations)

        # eta_sec = eta_sec * (
        #     self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        # )

        eta_sec = eta_sec * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        if self.mode == "train":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "time_data": self.iter_timer.data_time,
                "time_forward": self.iter_timer.forward_time,
                "time_loss": self.iter_timer.loss_time,
                "time_backward": self.iter_timer.backward_time,
                "mode": self.mode,
                "loss": self.loss.get_win_median(),
                "lr": self.lr,
            }
            if du.is_master_proc():
                # Iter
                iter_idx = cur_epoch * self.overall_iters + cur_iter + 1

                # Time:
                writer.add_scalar("Time/train_diff", stats["time_diff"], iter_idx)
                writer.add_scalar("Time/train_loss", stats["time_loss"], iter_idx)
                writer.add_scalar("Time/train_data", stats["time_data"], iter_idx)
                writer.add_scalar("Time/train_forward", stats["time_forward"], iter_idx)
                writer.add_scalar(
                    "Time/train_backward", stats["time_backward"], iter_idx
                )
                # LR
                writer.add_scalar("Utils/train_lr", stats["lr"], iter_idx)
                # Loss
                writer.add_scalar("Loss/train_loss", stats["loss"], iter_idx)

        elif self.mode == "val":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "mode": self.mode,
            }
            if du.is_master_proc():
                # Iter
                iter_idx = cur_epoch * self.overall_iters + cur_iter + 1

                # Time:
                writer.add_scalar("Time/val_diff", stats["time_diff"], iter_idx)

        elif self.mode == "test":
            stats = {
                "_type": "{}_iter".format(self.mode),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": self.iter_timer.seconds(),
                "mode": self.mode,
            }

            if du.is_master_proc():
                # Iter
                iter_idx = cur_iter + 1
                # Time:
                # writer.add_scalar("Time/test_diff", stats["time_diff"], iter_idx)

        else:
            raise NotImplementedError("Unknown mode: {}".format(self.mode))

        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def iter_data_toc(self):
        """
        Stop to record data processing time
        """
        self.iter_timer.data_toc()

    def iter_forward_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.forward_toc()

    def iter_loss_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.loss_toc()

    def iter_backward_toc(self):
        """
        Stop to record network backward time
        """
        self.iter_timer.backward_toc()

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()

        self.all_preds = []
        self.all_ori_boxes = []
        self.all_metadata = []

    def update_stats(self, preds, ori_boxes, metadata, loss=None, lr=None):
        """
        Update the current stats.
        Args:
            preds (tensor): prediction embedding.
            ori_boxes (tensor): original boxes (x1, y1, x2, y2).
            metadata (tensor): metadata of the AVA data.
            loss (float): loss value.
            lr (float): learning rate.
        """
        if self.mode in ["val", "test"]:
            self.all_preds.append(preds)
            self.all_ori_boxes.append(ori_boxes)
            self.all_metadata.append(metadata)
        if loss is not None:
            self.loss.add_value(loss)
        if lr is not None:
            self.lr = lr

    def finalize_metrics(self, log=True):
        """
        Calculate and log the final AVA metrics.
        """
        all_preds = torch.cat(self.all_preds, dim=0)
        all_ori_boxes = torch.cat(self.all_ori_boxes, dim=0)
        all_metadata = torch.cat(self.all_metadata, dim=0)

        if self.mode == "test" or (self.full_ava_test and self.mode == "val"):
            groundtruth = self.full_groundtruth
        else:
            groundtruth = self.mini_groundtruth

        self.full_map = evaluate_ava(
            all_preds,
            all_ori_boxes,
            all_metadata.tolist(),
            self.excluded_keys,
            self.class_whitelist,
            self.categories,
            groundtruth=groundtruth,
            video_idx_to_name=self.video_idx_to_name,
            out_folder=self.cfg.OUTPUT_DIR,
        )
        if log:
            stats = {"mode": self.mode, "map": self.full_map}
            logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        if self.mode in ["val", "test"]:
            self.finalize_metrics(log=False)
            stats = {
                "_type": "{}_epoch".format(self.mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "mode": self.mode,
                "map": self.full_map,
            }
            logging.log_json_stats(stats)
            if du.is_master_proc():
                # Add tensorboard metrix for visualization
                writer.add_scalar(
                    "Metric/{}_mAP".format(self.mode), self.full_map, cur_epoch + 1
                )


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_labels = torch.zeros((num_videos)).long()
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            self.video_labels[vid_id] = labels[ind]
            self.video_preds[vid_id] += preds[ind]
            self.clip_count[vid_id] += 1

    def get_current_accuracy(self, ks=(1, 5)):
        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [(x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct]
        assert len({len(ks), len(topks)}) == 1

        return topks

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        self.iter_timer._past_durations.append(self.iter_timer.seconds())
        try:
            eta_sec = mean(self.iter_timer._past_durations[-500:])
        except Exception as e:
            print(e)
            eta_sec = mean(self.iter_timer._past_durations)

        eta_sec = eta_sec * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        top1_acc, top5_acc = self.get_current_accuracy()

        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "top1_acc": "{:.{prec}f}".format(top1_acc, prec=2),
            "top5_acc": "{:.{prec}f}".format(top5_acc, prec=2),
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics(self, ks=(1, 5), out_folder=""):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(self.clip_count, self.num_clips)
            )
            logger.warning(self.clip_count)

        num_topks_correct = metrics.topks_correct(
            self.video_preds, self.video_labels, ks
        )
        topks = [(x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct]
        assert len({len(ks), len(topks)}) == 1
        stats = {"split": "test_final"}
        acc_dict = metrics.top1_per_class(self.video_preds, self.video_labels)

        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
            acc_dict["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
        logging.log_json_stats(stats)

        with open(os.path.join(out_folder, "acc_per_class.json"), "w") as f:
            json.dump(acc_dict, f)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def iter_data_toc(self):
        """
        Stop to record data processing time
        """
        self.iter_timer.data_toc()

    def iter_forward_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.forward_toc()

    def iter_loss_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.loss_toc()

    def iter_backward_toc(self):
        """
        Stop to record network backward time
        """
        self.iter_timer.backward_toc()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter, writer):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
            writer (tensorboard summarywriter): writer to storage the scalars for curve.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        # Method-1
        # eta_sec = self.iter_timer.seconds() * (
        #     self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        # )
        # Method-2
        # total_iters = cur_epoch * self.epoch_iters + cur_iter + 1
        # past_seconds = self.iter_timer.past()
        # eta_sec = (past_seconds / total_iters) * (
        #     self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        # )
        # Method-3
        self.iter_timer._past_durations.append(self.iter_timer.seconds())
        try:
            eta_sec = mean(self.iter_timer._past_durations[-500:])
        except Exception as e:
            print(e)
            eta_sec = mean(self.iter_timer._past_durations)

        eta_sec = eta_sec * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )

        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # import pdb; pdb.set_trace()

        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "time_data": self.iter_timer.data_time,
            "time_forward": self.iter_timer.forward_time,
            "time_loss": self.iter_timer.loss_time,
            "time_backward": self.iter_timer.backward_time,
            "eta": eta,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

        if du.is_master_proc():
            # Add metrics into tensorboard
            iter_idx = cur_epoch * self.epoch_iters + cur_iter + 1
            # Time
            writer.add_scalar("Time/train_diff", stats["time_diff"], iter_idx)
            writer.add_scalar("Time/train_loss", stats["time_loss"], iter_idx)
            writer.add_scalar("Time/train_forward", stats["time_forward"], iter_idx)
            writer.add_scalar("Time/train_backward", stats["time_backward"], iter_idx)
            # writer.add_scalar("Time/train_eta", stats['eta'], iter_idx)
            # Error
            writer.add_scalar("Error/train_top1", stats["top1_err"], iter_idx)
            writer.add_scalar("Error/train_top5", stats["top5_err"], iter_idx)
            # LR
            writer.add_scalar("Utils/train_lr", stats["lr"], iter_idx)
            writer.add_scalar("Utils/train_mem", stats["mem"], iter_idx)
            # Loss
            writer.add_scalar("Loss/train_loss", stats["loss"], iter_idx)

    def log_epoch_stats(self, cur_epoch, writer, ext_items=None):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err": top1_err,
            "top5_err": top5_err,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }

        if ext_items:
            stats.update(ext_items)  # expect dict

        logging.log_json_stats(stats)

        if du.is_master_proc():
            # Add tensorboard metrix for visualization
            writer.add_scalar("Epoch/train_loss", stats["loss"], cur_epoch + 1)
            writer.add_scalar("Epoch/train_top1_err", stats["top1_err"], cur_epoch + 1)
            writer.add_scalar("Epoch/train_top5_err", stats["top5_err"], cur_epoch + 1)
            if ext_items:
                for k in ext_items:
                    writer.add_scalar("Epoch/" + k, ext_items[k], cur_epoch)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err": self.mb_top1_err.get_win_median(),
            "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "top1_err": top1_err,
            "top5_err": top5_err,
            "min_top1_err": self.min_top1_err,
            "min_top5_err": self.min_top5_err,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        if du.is_master_proc() and writer is not None:
            writer.add_scalar("Epoch/val_top1_err", stats["top1_err"], cur_epoch)
            writer.add_scalar("Epoch/val_top5_err", stats["top5_err"], cur_epoch)


class MetaValMeter(object):
    """
    Measures validation stats for meta learnign setting.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        # self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        # self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        # self.mb_top5_err.reset()
        self.num_top1_mis = 0
        # self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        # self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        # self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "curr_top1_err": self.mb_top1_err.get_win_median(),
            "overall_top1_err": self.num_top1_mis / self.num_samples,
            # "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch, writer):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        top1_err = self.num_top1_mis / self.num_samples
        # top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        # self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "top1_err": top1_err,
            "top1_acc": 100.0 - top1_err,
            # "top5_err": top5_err,
            "min_top1_err": self.min_top1_err,
            # "min_top5_err": self.min_top5_err,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        if du.is_master_proc() and writer is not None:
            writer.add_scalar("Epoch/val_top1_err", stats["top1_err"], cur_epoch)
            # writer.add_scalar("Epoch/val_top5_err", stats["top5_err"], cur_epoch)


class MetaTestMeter(object):
    """
    Measures validation stats for meta learnign setting.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        # self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        # self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        # self.mb_top5_err.reset()
        self.num_top1_mis = 0
        # self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        # self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        # self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "test_iter",
            "epoch": self._cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1],
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "top1_err_curr": self.mb_top1_err.get_win_median(),
            "top1_err_overall": self.num_top1_mis / self.num_samples,
            # "top5_err": self.mb_top5_err.get_win_median(),
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        top1_err = self.num_top1_mis / self.num_samples
        # top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        # self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "test_epoch",
            "epoch": self._cfg.TEST.CHECKPOINT_FILE_PATH.split("/")[-1],
            "time_diff": self.iter_timer.seconds(),
            "final_top1_err": top1_err,
            "final_top1_acc": 100.0 - top1_err,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)
        return stats
        # if du.is_master_proc() and writer is not None:
        #     writer.add_scalar("Epoch/test_top1_err", stats["top1_err"], cur_epoch)
        # writer.add_scalar("Epoch/val_top5_err", stats["top5_err"], cur_epoch)


class MetaTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_meta_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_sem_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_meta_top1_mis = 0
        self.num_sem_top1_mis = 0
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_meta_top1_err.reset()
        self.mb_sem_top1_err.reset()
        self.num_meta_top1_mis = 0
        self.num_sem_top1_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def iter_data_toc(self):
        """
        Stop to record data processing time
        """
        self.iter_timer.data_toc()

    def iter_forward_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.forward_toc()

    def iter_loss_toc(self):
        """
        Stop to record network forward time
        """
        self.iter_timer.loss_toc()

    def iter_backward_toc(self):
        """
        Stop to record network backward time
        """
        self.iter_timer.backward_toc()

    def update_stats(self, meta_top1_err, sem_top1_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_meta_top1_err.add_value(meta_top1_err)
        self.mb_sem_top1_err.add_value(sem_top1_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_meta_top1_mis += meta_top1_err * mb_size
        self.num_sem_top1_mis += sem_top1_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter, writer):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
            writer (tensorboard summarywriter): writer to storage the scalars for curve.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        # Method-1
        # eta_sec = self.iter_timer.seconds() * (
        #     self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        # )
        # Method-2
        # total_iters = cur_epoch * self.epoch_iters + cur_iter + 1
        # past_seconds = self.iter_timer.past()
        # eta_sec = (past_seconds / total_iters) * (
        #     self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        # )
        # Method-3
        self.iter_timer._past_durations.append(self.iter_timer.seconds())
        try:
            eta_sec = mean(self.iter_timer._past_durations[-500:])
        except Exception as e:
            print(e)
            eta_sec = mean(self.iter_timer._past_durations)

        eta_sec = eta_sec * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )

        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        # import pdb; pdb.set_trace()

        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "time_data": self.iter_timer.data_time,
            "time_forward": self.iter_timer.forward_time,
            "time_loss": self.iter_timer.loss_time,
            "time_backward": self.iter_timer.backward_time,
            "eta": eta,
            "meta_top1_err": self.mb_meta_top1_err.get_win_median(),
            "sem_top1_err": self.mb_sem_top1_err.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        logging.log_json_stats(stats)

        if du.is_master_proc():
            # Add metrics into tensorboard
            iter_idx = cur_epoch * self.epoch_iters + cur_iter + 1
            # Time
            writer.add_scalar("Time/train_diff", stats["time_diff"], iter_idx)
            writer.add_scalar("Time/train_loss", stats["time_loss"], iter_idx)
            writer.add_scalar("Time/train_forward", stats["time_forward"], iter_idx)
            writer.add_scalar("Time/train_backward", stats["time_backward"], iter_idx)
            # writer.add_scalar("Time/train_eta", stats['eta'], iter_idx)
            # Error
            writer.add_scalar(
                "Error/train_meta_top1_err", stats["meta_top1_err"], iter_idx
            )
            writer.add_scalar(
                "Error/train_sem_top1_err", stats["sem_top1_err"], iter_idx
            )
            # LR
            writer.add_scalar("Utils/train_lr", stats["lr"], iter_idx)
            writer.add_scalar("Utils/train_mem", stats["mem"], iter_idx)
            # Loss
            writer.add_scalar("Loss/train_loss", stats["loss"], iter_idx)

    def log_epoch_stats(self, cur_epoch, writer, ext_items=None):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
            writer (tensorboard summarywriter): writer to storage the scalars for curve
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        meta_top1_err = self.num_meta_top1_mis / self.num_samples
        sem_top1_err = self.num_sem_top1_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "meta_top1_err": meta_top1_err,
            "sem_top1_err": sem_top1_err,
            "loss": avg_loss,
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }

        if ext_items:
            stats.update(ext_items)  # expect dict

        logging.log_json_stats(stats)

        if du.is_master_proc():
            # Add tensorboard metrix for visualization
            writer.add_scalar("Epoch/train_loss", stats["loss"], cur_epoch + 1)
            writer.add_scalar(
                "Epoch/train_meta_top1_err", stats["meta_top1_err"], cur_epoch + 1
            )
            writer.add_scalar(
                "Epoch/train_sem_top1_err", stats["sem_top1_err"], cur_epoch + 1
            )
            if ext_items:
                for k in ext_items:
                    writer.add_scalar("Epoch/" + k, ext_items[k], cur_epoch)
