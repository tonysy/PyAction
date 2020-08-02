#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import sys

sys.path.insert(0, ".")

import os
import numpy as np
import torch
# from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
from pyaction.utils.precise_bn_fewshot import get_bn_modules, update_bn_stats

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

    for cur_iter, (support_x, support_y, target_x, target_y) in enumerate(train_loader):  # data_dict

        # tensor size checked.
        # support_x: (batchsize, classes_per_set * samples_per_class, 3, 8, 224, 224)
        # support_y: (batchsize, classes_per_set * samples_per_class)
        # target_x:  (batchsize, samples_per_class, 3, 8, 224, 224)
        # target_y:  (batchsize, samples_per_class)

        # support_y: Add extra dimension for the one_hot
        support_y = torch.unsqueeze(support_y, 2)  # (batchsize, classes_per_set * samples_per_class, 1)
        batch_size = support_y.size()[0]
        n_samples = support_y.size()[1]
        support_y_one_hot = torch.zeros(batch_size, n_samples, cfg.FEW_SHOT.CLASSES_PER_SET)  # the last dim as one-hot
        support_y_one_hot.scatter_(2, support_y, 1.0)

        # Transfer the data to the current GPU device.
        support_x = support_x.cuda(non_blocking=True)
        support_y_one_hot = support_y_one_hot.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)
        
        # Record data time
        train_meter.iter_data_toc()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            # preds = model(inputs, meta["boxes"])
            pass
        else:
            # Perform the forward pass.
            # preds = model(inputs)
            # preds = model(support_x, support_y_one_hot, target_x)
            acc, loss = model(support_x, support_y_one_hot, target_x, target_y)

        # import pdb; pdb.set_trace()

        # Record network forward time
        train_meter.iter_forward_toc()

        # # Explicitly declare reduction to mean.
        # loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # # Compute the loss.
        # loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Record loss computation time
        train_meter.iter_loss_toc()

        if hasattr(cfg, "NO_TRAIN") and cfg.NO_TRAIN:
            pass
        else:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
            
        # Record network backward time
        train_meter.iter_backward_toc()

        if cfg.DETECTION.ENABLE:
            # if cfg.NUM_GPUS > 1:
            #     loss = du.all_reduce([loss])[0]
            # loss = loss.item()

            # train_meter.iter_toc()
            # # Update and log stats.
            # train_meter.update_stats(None, None, None, loss, lr)
            pass
        else:
            # # Compute the errors.
            # num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            # top1_err, top5_err = [
            #     (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            # ]

            # # Gather all the predictions across all the devices.
            # if cfg.NUM_GPUS > 1:
            #     loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

            # # Copy the stats from GPU to CPU (sync point).
            # loss, top1_err, top5_err = (
            #     loss.item(),
            #     top1_err.item(),
            #     top5_err.item(),
            # )

            # train_meter.iter_toc()
            # # Update and log stats.
            # train_meter.update_stats(
            #     top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
            # )

            # print("before reduce: {}".format(acc))

            # Gather all the predictions across all the devices.
            if cfg.NUM_GPUS > 1:
                loss, acc = du.all_reduce([loss, acc])

            # print("after reduce: {}".format(acc))
            # it works like, e.g.
            # before reduce: 1.0
            # before reduce: 0.0
            # before reduce: 0.0
            # before reduce: 0.0
            # after reduce: 0.25
            # after reduce: 0.25
            # after reduce: 0.25
            # after reduce: 0.25

            # Copy the stats from GPU to CPU (sync point).
            loss, acc = (
                loss.item(),
                acc.item(),
            )

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(
                1.0-acc, .0, loss, lr, support_x.size(0) * cfg.NUM_GPUS
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

    for cur_iter, (support_x, support_y, target_x, target_y) in enumerate(val_loader):
        
        # support_y: Add extra dimension for the one_hot
        support_y = torch.unsqueeze(support_y, 2)  # (batchsize, classes_per_set * samples_per_class, 1)
        batch_size = support_y.size()[0]
        n_samples = support_y.size()[1]
        support_y_one_hot = torch.zeros(batch_size, n_samples, cfg.FEW_SHOT.CLASSES_PER_SET)  # the last dim as one-hot
        support_y_one_hot.scatter_(2, support_y, 1.0)

        # Transfer the data to the current GPU device.
        support_x = support_x.cuda(non_blocking=True)
        support_y_one_hot = support_y_one_hot.cuda(non_blocking=True)
        target_x = target_x.cuda(non_blocking=True)
        target_y = target_y.cuda(non_blocking=True)

        # # Transfer the data to the current GPU device.
        # if isinstance(inputs, (list,)):
        #     for i in range(len(inputs)):
        #         inputs[i] = inputs[i].cuda(non_blocking=True)
        # else:
        #     inputs = inputs.cuda(non_blocking=True)
        # labels = labels.cuda()
        # for key, val in meta.items():
        #     if isinstance(val, (list,)):
        #         for i in range(len(val)):
        #             val[i] = val[i].cuda(non_blocking=True)
        #     else:
        #         meta[key] = val.cuda(non_blocking=True)

        if cfg.DETECTION.ENABLE:
            # # Compute the predictions.
            # preds = model(inputs, meta["boxes"])

            # preds = preds.cpu()
            # ori_boxes = meta["ori_boxes"].cpu()
            # metadata = meta["metadata"].cpu()

            # if cfg.NUM_GPUS > 1:
            #     preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
            #     ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
            #     metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            # val_meter.iter_toc()
            # # Update and log stats.
            # val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
            pass
        else:
            # preds = model(inputs)
            acc, _ = model(support_x, support_y_one_hot, target_x, target_y)

            # # Compute the errors.
            # num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

            # # Combine the errors across the GPUs.
            # top1_err, top5_err = [
            #     (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            # ]

            if cfg.NUM_GPUS > 1:
                acc = du.all_reduce([acc])

            # Copy the errors from GPU to CPU (sync point).
            if isinstance(acc, list):
                acc = acc[0].item()
            else:
                acc = acc.item()

            val_meter.iter_toc()

            # Update and log stats.
            val_meter.update_stats(1.0-acc, .0, support_x.size(0) * cfg.NUM_GPUS)

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

    num_classes = cfg.FEW_SHOT.CLASSES_PER_SET
    def _gen_loader():
        for support_x, support_y, target_x, _ in loader:

            support_y = torch.unsqueeze(support_y, 2)  # (batchsize, classes_per_set * samples_per_class, 1)
            batch_size = support_y.size()[0]
            n_samples = support_y.size()[1]
            support_y_one_hot = torch.zeros(batch_size, n_samples, num_classes)  # the last dim as one-hot
            support_y_one_hot.scatter_(2, support_y, 1.0)

            support_x = support_x.cuda(non_blocking=True)
            support_y_one_hot = support_y_one_hot.cuda(non_blocking=True)
            target_x = target_x.cuda(non_blocking=True)
            yield support_x, support_y_one_hot, target_x

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
            model.module.g if hasattr(model, "module") else model.g,
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
                # len(train_loader) == len(dataset)//ngpus
                train_loader, model, cfg, len(train_loader)//5  #cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg)
            
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
