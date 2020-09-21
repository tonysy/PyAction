#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import sys

sys.path.insert(0, ".")

import os
import numpy as np
import torch
import argparse

import pyaction.utils.checkpoint as cu
import pyaction.utils.distributed as du
import pyaction.utils.logging as logging
import pyaction.utils.misc as misc
from pyaction.datasets import loader

# from pyaction.models import model_builder
from pyaction.utils.meters import AVAMeter, TestMeter
from net import build_model

import pyaction.utils.multiprocessing as mpu
from config import config
from tqdm import tqdm

def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide PyAction training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards", help="Number of shards using by the job", default=1, type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    # parser.add_argument(
    #     "--cfg",
    #     dest="cfg_file",
    #     help="Path to the config file",
    #     default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
    #     type=str,
    # )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    # cfg = get_cfg()
    cfg = config
    # Load config from cfg.
    # if args.cfg_file is not None:
    #     cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def perform_test(test_loader, model, test_meter, cfg):
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
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    # debug
    acc_list = []

    bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
    pbar = tqdm(test_loader, bar_format=bar_format)

    for cur_iter, (support_x, support_y, target_x, target_y) in enumerate(pbar):
    # for cur_iter, (support_x, support_y, target_x, target_y) in enumerate(test_loader):

        # print("iter {}/{}:".format(cur_iter, len(test_loader)))

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

            # test_meter.iter_toc()
            # # Update and log stats.
            # test_meter.update_stats(
            #     preds.detach().cpu(), ori_boxes.detach().cpu(), metadata.detach().cpu(),
            # )
            # test_meter.log_iter_stats(None, cur_iter)
            pass
        else:
            # preds = model(inputs)
            acc, _ = model(support_x, support_y_one_hot, target_x, target_y)

            if cfg.NUM_GPUS > 1:
                acc = du.all_reduce([acc])

            # Copy the errors from GPU to CPU (sync point).
            if isinstance(acc, list):
                acc = acc[0].item()
            else:
                acc = acc.item()

            # print(acc)

            # debug
            acc_list.append(acc)

            test_meter.iter_toc()

            # # Gather all the predictions across all the devices to perform ensemble.
            # if cfg.NUM_GPUS > 1:
            #     preds, labels, video_idx = du.all_gather([preds, labels, video_idx])

            # test_meter.iter_toc()
            # # Update and log stats.
            # test_meter.update_stats(
            #     preds.detach().cpu(), labels.detach().cpu(), video_idx.detach().cpu(),
            # )
            # test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    # if cfg.DETECTION.ENABLE:
    #     test_meter.finalize_metrics()
    # else:
    #     test_meter.finalize_metrics(out_folder=cfg.OUTPUT_DIR)

    test_meter.reset()

    # debug
    return sum(acc_list)/len(acc_list)


def test(cfg):
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
    # logger = logging.setup_logging(cfg.OUTPUT_DIR)
    logger = logging.setup_logging(os.path.join(cfg.OUTPUT_DIR, "test_log.txt"))

    # logger = logging.get_logger(__name__)
    logger.info("Running with full config:\n{}".format(cfg))
    base_config = cfg.__class__.__base__()
    logger.info(
        "different config with base class:\n{}".format(cfg.show_diff(base_config))
    )
    is_master = du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS)
    if hasattr(cfg.TEST, "LOAD_EPOCH") and cfg.TEST.LOAD_EPOCH:
        for cur_epoch in cfg.TEST.EPOCH_IDS:
            for cur_split in cfg.TEST.SPLITS:
                if hasattr(cfg.TEST, "ENDLESS") and cfg.TEST.ENDLESS \
                    and cur_epoch == cfg.TEST.EPOCH_IDS[-1] and cur_split == cfg.TEST.SPLITS[-1]:  # last iteration
                    cfg.TEST.NTEST = 1000
                cfg.TEST.CUR_EPOCH = cur_epoch
                cfg.TEST.CUR_SPLIT = cur_split
                test_model(cfg, is_master)
    else:
        for cur_split in cfg.TEST.SPLITS:
            if hasattr(cfg.TEST, "ENDLESS") and cfg.TEST.ENDLESS \
                and cur_split == cfg.TEST.SPLITS[-1]:  # last iteration
                cfg.TEST.NTEST = 1000
            cfg.TEST.CUR_SPLIT = cur_split
            test_model(cfg, is_master)

    
# Test on single model
def test_model(cfg, is_master):

    # Create the log folder
    if is_master:
        logpath = os.path.join(cfg.OUTPUT_DIR, "inference_logs")
        if not os.path.exists(logpath):
            os.mkdir(logpath)

    # Set current logger
    logfilename = cfg.TEST.CUR_EPOCH if hasattr(cfg.TEST, "LOAD_EPOCH") and cfg.TEST.LOAD_EPOCH else "xxx"
    logger = logging.setup_logging(
        os.path.join(
            cfg.OUTPUT_DIR, 
            "inference_logs",
            "{}_{}.log".format(cfg.TEST.CUR_SPLIT, logfilename)
        )
    )

    # LOG VARS #
    model_name = ""
    # Unified test metric
    unified_eval = hasattr(cfg.TEST, "UNIFIED_EVAL") and cfg.TEST.UNIFIED_EVAL
    # Center-crop & multi-view
    center_crop_multi_view = hasattr(cfg.TEST, "CENTER_CROP_MULTI_VIEW") and cfg.TEST.CENTER_CROP_MULTI_VIEW
    if unified_eval:
        eval_mode = "unified eval"
    elif center_crop_multi_view:
        eval_mode = "center-crop multi-view"
    else:
        eval_mode = "multi-crop multi-view"
    # Conflit
    assert not (unified_eval and center_crop_multi_view)

    # Build the video model and print model statistics.
    model = build_model(cfg)  # model = model_builder.build_model(cfg)

    # if du.is_master_proc():
    #     misc.log_model_info(model, cfg, is_train=False)

    ### If specified, either file path or epoch id is specified. ###
    assert (not cfg.TEST.CHECKPOINT_FILE_PATH) or not (hasattr(cfg.TEST, "LOAD_EPOCH") and cfg.TEST.LOAD_EPOCH)

    # Load a checkpoint to test if applicable.
    if hasattr(cfg.TEST, "LOAD_CLASSIFIER") and cfg.TEST.LOAD_CLASSIFIER:
        logger.info("Load from the {}-th epoch...".format(cfg.TEST.CUR_EPOCH))
        cpath = cu.get_checkpoint(cfg.OUTPUT_DIR, cfg.TEST.CUR_EPOCH)
        cu.load_checkpoint(
            cpath, 
            model.module.g if hasattr(model, "module") else model.g, 
            cfg.NUM_GPUS > 1,
            strict=False,
        )
        model_name = cpath  # for print

    elif cfg.TEST.CHECKPOINT_FILE_PATH != "":
        logger.info("Loading from given file {}...".format(cfg.TEST.CHECKPOINT_FILE_PATH))
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model.module.g if hasattr(model, "module") else model.g,
            cfg.NUM_GPUS > 1,
            None,
            inflation=True,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
        model_name = cfg.TEST.CHECKPOINT_FILE_PATH  # for print
    elif hasattr(cfg.TEST, "LOAD_EPOCH") and cfg.TEST.LOAD_EPOCH:
        logger.info("Load from the {}-th epoch...".format(cfg.TEST.CUR_EPOCH))
        cpath = cu.get_checkpoint(cfg.OUTPUT_DIR, cfg.TEST.CUR_EPOCH)
        cu.load_checkpoint(cpath, model, cfg.NUM_GPUS > 1)
        model_name = cpath  # for print
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Loading from last checkpoint...")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
        model_name = last_checkpoint  # for print
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        model_name = cfg.TRAIN.CHECKPOINT_FILE_PATH  # for print
    else:
        # raise NotImplementedError("Unknown way to load checkpoint.")
        logger.info("Testing with random initialization. Only for debugging.")

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        # assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE
        # test_meter = AVAMeter(len(test_loader), cfg, mode="test")
        pass
    else:
        if unified_eval:
            num_clips = 1
        elif center_crop_multi_view:
            num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * 1
        else:
            num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
        
        assert num_clips in [1, 10, 30]
        assert (len(test_loader.dataset) % (num_clips) == 0)
        
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            len(test_loader.dataset) // num_clips,
            num_clips,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
        )

    # # Perform multi-view test on the entire dataset.
    n_test = cfg.TEST.NTEST if hasattr(cfg.TEST, "NTEST") else 1000
    results = []
    split_name = cfg.TEST.CUR_SPLIT

    for i in range(n_test):
        assert len(results) == i
        results.append(perform_test(test_loader, model, test_meter, cfg))
        if is_master:
            m0 = "-------------------epoch {}/{}:------------------------".format(i, n_test)
            output_messages = [
                m0,
                "                   split: {}".format(split_name+".csv"),
                "               rand seed: {}".format(cfg.RNG_SEED),
                "               eval mode: {}".format(eval_mode),
                "         model evaluated: {}".format(model_name),
                "           cur epoch acc: {}".format(results[-1]),
                "   cur max, min, maxdiff: {}, {}, {}".format(max(results), min(results), max(results)-min(results)),
                "             cur avg acc: {}".format(sum(results)/len(results)),
                "-" * len(m0),
            ]
            for m in output_messages:
                # print(m)
                logger.info(m)

            # l0 = "-------------------epoch {}/{}:------------------------".format(i, n_test)
            # print(l0)
            # print("                   split: {}".format(split_name+".csv"))
            # print("               rand seed: {}".format(cfg.RNG_SEED))
            # print("               eval mode: {}".format(eval_mode))
            # print("         model evaluated: {}".format(model_name))
            # print("           cur epoch acc: {}".format(results[-1]))
            # print("   cur max, min, maxdiff: {}, {}, {}".format(max(results), min(results), max(results)-min(results)))
            # print("             cur avg acc: {}".format(sum(results)/len(results)))
            # print("-" * len(l0))

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg.link_log()
    print("soft link to {}".format(cfg.OUTPUT_DIR))

    # Perform multi-clip testing.
    if cfg.NUM_GPUS > 1 and cfg.DIST_MULTIPROCESS:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                test,
                args.init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    elif cfg.NUM_GPUS == 1:
        test(cfg=cfg)
    else:
        try:
            torch.distributed.init_process_group(
                backend=cfg.DIST_BACKEND,
                init_method=args.init_method,
                world_size=cfg.NUM_SHARDS,
                rank=cfg.SHARD_ID
            )
            test(cfg=cfg)

        except Exception as e:
            raise e


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()