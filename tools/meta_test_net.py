#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import sys

sys.path.insert(0, ".")

import os
import numpy as np
import torch
import time
import glob
import re
from pprint import pformat

import pyaction.utils.checkpoint as cu
import pyaction.utils.distributed as du
import pyaction.utils.logging as logging
from pyaction.datasets import loader
import pyaction.utils.metrics as metrics

# from pyaction.models import model_builder
from pyaction.utils.meters import MetaTestMeter
from net import build_model


@torch.no_grad()
def perform_test(model, cfg, logger):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        raise NotImplementedError
    else:
        sanity_check(cfg, test_loader)
        test_meter = MetaTestMeter(len(test_loader), cfg)

    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (
        support_data,
        support_meta_label,
        query_data,
        query_meta_label,
    ) in enumerate(test_loader):
        # (batchsize, classes_per_set * samples_per_class)
        batch_size, n_samples = support_meta_label.shape

        # Transfer the data to the current GPU device.
        support_data = support_data.cuda(non_blocking=True)
        support_meta_label = support_meta_label.cuda(non_blocking=True)

        query_data = query_data.cuda(non_blocking=True)
        query_meta_label = query_meta_label.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            raise NotImplementedError
        else:
            # preds = model(inputs)
            preds = model([support_data, query_data], support_meta_label)

            num_topks_correct = metrics.topks_correct(
                preds, query_meta_label.view(-1), (1, 5)
            )
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            test_meter.iter_toc()

            # Update and log stats.
            test_meter.update_stats(
                top1_err, top5_err, support_data.size(0) * cfg.NUM_GPUS
            )

            test_meter.log_iter_stats(-1, cur_iter)

        test_meter.iter_tic()

    # Log epoch stats.
    test_meter.log_epoch_stats(-1)
    test_meter.reset()


def filter_by_iters(file_list, start_epoch, end_epoch):
    # sort file_list by modified time
    file_list.sort(key=os.path.getmtime)

    if start_epoch is None:
        if end_epoch is None:
            # use latest ckpt if start_epoch and end_iter are not given
            return [file_list[-1]]
        else:
            start_epoch = 0
    elif end_epoch is None:
        end_epoch = float("inf")

    iter_infos = [re.split(r"checkpoint_epoch_|\.pyth", f)[-2] for f in file_list]
    keep_list = [0] * len(iter_infos)
    start_index = 0
    if "final" in iter_infos and iter_infos[-1] != "final":
        start_index = iter_infos.index("final")

    for i in range(len(iter_infos) - 1, start_index, -1):
        if iter_infos[i] == "final":
            if end_epoch == float("inf"):
                keep_list[i] = 1
        elif float(start_epoch) < float(iter_infos[i]) < float(end_epoch):
            keep_list[i] = 1
            if float(iter_infos[i - 1]) > float(iter_infos[i]):
                break

    return [filename for keep, filename in zip(keep_list, file_list) if keep == 1]


def get_valid_files(cfg, logger):

    # if cfg.TEST.CHECKPOINT_FILE_PATH != "":
    #     model_weights = TEST.CHECKPINT_FILE_PATH
    #     assert os.path.exists(model_weights), "{} not exist!!!".format(model_weights)
    #     return [model_weights]

    file_list = glob.glob(
        os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_*.pyth")
    )
    if len(file_list) == 0:
        assert len(file_list) != 0, "No valid file found on local"

    file_list = filter_by_iters(file_list, cfg.TEST.START_EPOCH, cfg.TEST.END_EPOCH)
    assert file_list, "No checkpoint valid in {}.".format(cfg.OUTPUT_DIR)
    logger.info(
        "All files below will be tested in order:\n{}".format(pformat(file_list))
    )
    return file_list


def meta_test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger = logging.setup_logging(
        os.path.join(cfg.OUTPUT_DIR, f"test_log_{timestamp}.log")
    )

    # logger = logging.get_logger(__name__)
    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info(
        "different config with base class:\n{}".format(cfg.show_diff(base_config))
    )

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        # Build the video model and print model statistics.
        model = build_model(cfg)  # model = model_builder.build_model(cfg)
        logger.info("Current checkpoint is:{}".format(cfg.TEST.CHECKPOINT_FILE_PATH))
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
        perform_test(model, cfg, logger)

    elif (
        cu.has_checkpoint(cfg.OUTPUT_DIR) and cfg.TEST.START_EPOCH == cfg.TEST.END_EPOCH
    ):
        # Build the video model and print model statistics.
        model = build_model(cfg)  # model = model_builder.build_model(cfg)
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        logger.info("Current checkpoint is:{}".format(last_checkpoint))
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
        perform_test(model, cfg, logger)

    elif cfg.TEST.START_EPOCH != cfg.TEST.END_EPOCH:
        file_list = get_valid_files(cfg, logger)
        for valid_file in file_list:
            # Build the video model and print model statistics.
            model = build_model(cfg)  # model = model_builder.build_model(cfg)
            logger.info("Current checkpoint is:{}".format(valid_file))
            cu.load_checkpoint(valid_file, model, cfg.NUM_GPUS > 1)
            perform_test(model, cfg, logger)
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # is_master = du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS)


def sanity_check(cfg, test_loader):
    unified_eval = cfg.TEST.UNIFIED_EVAL
    # Center-crop & multi-view
    center_crop_multi_view = cfg.TEST.CENTER_CROP_MULTI_VIEW
    # if unified_eval:
    #     eval_mode = "unified eval"
    # elif center_crop_multi_view:
    #     eval_mode = "center-crop multi-view"
    # else:
    #     eval_mode = "multi-crop multi-view"
    # Conflit
    assert not (unified_eval and center_crop_multi_view)

    if unified_eval:
        num_clips = 1
    elif center_crop_multi_view:
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * 1
    else:
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS

    assert num_clips in [1, 10, 30]
    assert len(test_loader.dataset) % (num_clips) == 0
