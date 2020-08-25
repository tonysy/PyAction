#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import sys

sys.path.insert(0, ".")

import os
import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from torch.utils.tensorboard import SummaryWriter
import pyaction.models.losses as losses
import pyaction.models.optimizer as optim
import pyaction.utils.checkpoint as cu
import pyaction.utils.distributed as du
import pyaction.utils.logging as logging
import pyaction.utils.metrics as metrics
import pyaction.utils.misc as misc
from pyaction.datasets import loader

# from pyaction.models import model_builder
from pyaction.utils.meters import AVAMeter, TrainMeter, ValMeter

from net import build_model


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (tensorboard summarywriter): writer to storage the scalars for curve
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)
        # Record data time
        train_meter.iter_data_toc()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
        else:
            # Perform the forward pass.
            preds = model(inputs)

        # Record network forward time
        train_meter.iter_forward_toc()
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Record loss computation time
        train_meter.iter_loss_toc()

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        # Record network backward time
        train_meter.iter_backward_toc()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            # Copy the stats from GPU to CPU (sync point).
            loss, top1_err, top5_err = (
                loss.item(),
                top1_err.item(),
                top5_err.item(),
            )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter, writer)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, writer)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (tensorboard summarywriter): writer to storage the scalars for curve
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        labels = labels.cuda()
        for key, val in meta.items():
            if isinstance(val, (list,)):
                for i in range(len(val)):
                    val[i] = val[i].cuda(non_blocking=True)
            else:
                meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])

            preds = preds.cpu()
            ori_boxes = meta["ori_boxes"].cpu()
            metadata = meta["metadata"].cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
        else:
            preds = model(inputs)

            # Compute the errors.
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # Combine the errors across the GPUs.
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])

            # Copy the errors from GPU to CPU (sync point).
            top1_err, top5_err = top1_err.item(), top5_err.item()

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS)

        if cfg.DETECTION.ENABLE:
            val_meter.log_iter_stats(cur_epoch, cur_iter, writer)
        else:
            val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch, writer)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logger = logging.setup_logging(os.path.join(cfg.OUTPUT_DIR, "train_log.txt"))

    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info(
        "different config with base class:\n{}".format(cfg.show_diff(base_config))
    )

    # logger = logging.get_logger(name='pyaction')
    # logger = logging.get_logger(__name__)

    # Print config.
    # logger.info("Train with config:")
    # logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    # model = model_builder.build_model(cfg)
    model = build_model(cfg)
    if du.is_master_proc():
        # print(type(model))
        if cfg.NUM_GPUS > 1:
            model_plain = model.module
        else:
            model_plain = model
        writer = SummaryWriter(cfg.OUTPUT_DIR)  # , **kwargs)

        misc.log_model_info(model_plain, cfg, is_train=True, writer=writer)
    else:
        writer = None

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    if cfg.DETECTION.ENABLE:
        train_meter = AVAMeter(len(train_loader), cfg, mode="train")
        val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    else:
        train_meter = TrainMeter(len(train_loader), cfg)
        val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, min(len(train_loader)//2, cfg.BN.NUM_BATCHES_PRECISE)
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
