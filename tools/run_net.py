#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
import sys

sys.path.insert(0, ".")

import argparse
import torch

import pyaction.utils.checkpoint as cu
import pyaction.utils.multiprocessing as mpu

from config import config
from test_net import test

if hasattr(config, "FEW_SHOT"):
    from train_net_fewshot import train
else:
    from train_net import train

# from pyaction.config.defaults import get_cfg


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


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    cfg = load_config(args)
    cfg.link_log()
    print("soft link to {}".format(cfg.OUTPUT_DIR))

    # Perform training.
    if cfg.TRAIN.ENABLE:
        if cfg.NUM_GPUS > 1 and cfg.DIST_MULTIPROCESS:
            torch.multiprocessing.spawn(
                mpu.run,
                nprocs=cfg.NUM_GPUS,
                args=(
                    cfg.NUM_GPUS,
                    train,
                    args.init_method,
                    cfg.SHARD_ID,
                    cfg.NUM_SHARDS,
                    cfg.DIST_BACKEND,
                    cfg,
                ),
                daemon=False,
            )
        elif cfg.NUM_GPUS == 1:
            train(cfg=cfg)
        else:
            try:
                torch.distributed.init_process_group(
                    backend=cfg.DIST_BACKEND,
                    init_method=args.init_method,
                    world_size=cfg.NUM_SHARDS,
                    rank=cfg.SHARD_ID
                )
                train(cfg=cfg)

            except Exception as e:
                raise e

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
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
