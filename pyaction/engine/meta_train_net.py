#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import sys

sys.path.insert(0, ".")

import os
import time
import numpy as np
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from torch.utils.tensorboard import SummaryWriter

import pyaction.models.optimizer as optim
import pyaction.utils.checkpoint as cu
import pyaction.utils.distributed as du
import pyaction.utils.logging as logging
import pyaction.utils.metrics as metrics
import pyaction.utils.misc as misc
from pyaction.datasets import loader

from pyaction.utils.meters import AVAMeter, MetaValMeter, MetaTrainMeter

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

    # tmp = train_loader.dataset[0]

    for cur_iter, (
        support_data,
        support_meta_label,
        support_sem_label,
        query_data,
        query_meta_label,
        query_sem_label,
    ) in enumerate(train_loader):

        # (batchsize, classes_per_set * samples_per_class)
        batch_size, n_samples = support_meta_label.shape

        # Transfer the data to the current GPU device.
        support_data = support_data.cuda(non_blocking=True)
        support_meta_label = support_meta_label.cuda(non_blocking=True)
        support_sem_label = support_sem_label.cuda(non_blocking=True)

        query_data = query_data.cuda(non_blocking=True)
        query_meta_label = query_meta_label.cuda(non_blocking=True).view(-1)
        query_sem_label = query_sem_label.cuda(non_blocking=True)
        all_sem_labels = torch.cat([support_sem_label, query_sem_label], dim=1).view(-1)

        # Record data time
        train_meter.iter_data_toc()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            # preds = model(inputs, meta["boxes"])
            raise NotImplementedError
        else:
            # Perform the forward pass.
            # preds = model([support_data, query_data], support_meta_label)
            query_meta_preds, all_sem_preds, losses = model(
                [support_data, query_data],
                support_meta_label,
                query_meta_label,
                all_sem_labels,
            )

        # Record network forward time
        train_meter.iter_forward_toc()

        # # Explicitly declare reduction to mean.
        # loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # # Compute the loss.
        # loss = loss_fun(preds, query_meta_label)
        loss = sum(loss for key, loss in losses.items() if "loss" in key)

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
            raise NotImplementedError
        else:
            meta_num_correct = metrics.topks_correct(
                query_meta_preds, query_meta_label, (1,)
            )[0]
            meta_top1_err = (1.0 - meta_num_correct / query_meta_preds.size(0)) * 100.0

            sem_num_correct = metrics.topks_correct(
                all_sem_preds, all_sem_labels, (1,)
            )[0]
            sem_top1_err = (1.0 - sem_num_correct / all_sem_preds.size(0)) * 100.0

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, meta_top1_err, sem_top1_err = du.all_reduce(
                    [loss, meta_top1_err, sem_top1_err]
                )

            # Copy the stats from GPU to CPU (sync point).
            loss, meta_top1_err, sem_top1_err = (
                loss.item(),
                meta_top1_err.item(),
                sem_top1_err.item(),
            )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                meta_top1_err,
                sem_top1_err,
                loss,
                lr,
                support_data.size(0) * cfg.NUM_GPUS,
            )

        train_meter.log_iter_stats(cur_epoch, cur_iter, writer)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch, writer, ext_items=None)
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

    for cur_iter, (
        support_data,
        support_meta_label,
        support_sem_label,
        query_data,
        query_meta_label,
        query_sem_label,
    ) in enumerate(val_loader):
        # (batchsize, classes_per_set * samples_per_class)
        batch_size, n_samples = support_meta_label.shape

        # Transfer the data to the current GPU device.
        support_data = support_data.cuda(non_blocking=True)
        support_meta_label = support_meta_label.cuda(non_blocking=True)

        query_data = query_data.cuda(non_blocking=True)
        query_meta_label = query_meta_label.cuda(non_blocking=True)

        # Record data time
        # val_meter.iter_data_toc()

        if cfg.DETECTION.ENABLE:
            raise NotImplementedError
        else:
            # preds = model(inputs)
            query_meta_preds = model(
                [support_data, query_data],
                support_meta_label,
            )

            meta_num_correct = metrics.topks_correct(
                query_meta_preds, query_meta_label, (1,)
            )[0]
            meta_top1_err = (1.0 - meta_num_correct / query_meta_preds.size(0)) * 100.0

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                meta_top1_err, meta_top1_err = du.all_reduce(
                    [meta_top1_err, meta_top1_err]
                )

            meta_top1_err, meta_top1_err = meta_top1_err.item(), meta_top1_err.item()

            val_meter.iter_toc()

            # Update and log stats.
            val_meter.update_stats(
                meta_top1_err, meta_top1_err, support_data.size(0) * cfg.NUM_GPUS
            )

        if cfg.DETECTION.ENABLE:
            val_meter.log_iter_stats(cur_epoch, cur_iter, writer)
        else:
            val_meter.log_iter_stats(cur_epoch, cur_iter)

        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch, writer)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, cfg, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    # num_classes = cfg.FEW_SHOT.CLASSES_PER_SET

    def _gen_loader():
        # TODO: re-format this function
        pass

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def meta_train(cfg):
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

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger = logging.setup_logging(
        os.path.join(cfg.OUTPUT_DIR, f"train_log_{timestamp}.log")
    )

    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info(
        "different config with base class:\n{}".format(cfg.show_diff(base_config))
    )

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc():
        # TODO: add support for model information measurement
        # if cfg.NUM_GPUS > 1:
        #     model_plain = model.module
        # else:
        #     model_plain = model
        writer = SummaryWriter(cfg.OUTPUT_DIR)  # , **kwargs)

        # misc.log_model_info(model_plain, cfg, is_train=True, writer=writer)
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
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":  # for loading pretrain
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            # model.module.g if hasattr(model, "module") else model.g,
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
        train_meter = MetaTrainMeter(len(train_loader), cfg)
        val_meter = MetaValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        if cur_epoch == 0:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)

        # Train for one epoch.
        # BN will not update if cfg.FREEZE_BN is True
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, writer)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
