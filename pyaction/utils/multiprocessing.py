#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import torch
import os
import time


def _get_local_ip():
    """
    Find an ip address of current machine / node.
    """
    import ifcfg

    # ip = ifcfg.default_interface()["inet4"][0]
    ip = ifcfg.interfaces()["ib0"]["inet"]
    return ip


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def run(local_rank, num_proc, func, init_method, shard_id, num_shards, backend, cfg):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    if init_method == "auto":
        if rank > 0:
            try:
                for _ in range(600):
                    if os.path.exists(".ip_dist_url"):
                        break
                    time.sleep(1)
                with open(".ip_dist_url", "r") as f:
                    init_method = f.readline()
            except Exception:
                assert (
                    num_shards == 1
                ), "dist_url=auto cannot work with distributed training."
        else:
            port = _find_free_port()
            init_method = f"tcp://127.0.0.1:{port}"
            local_ip = _get_local_ip()
            ip_dist_url = f"tcp://{local_ip}:{port}"
            with open(".ip_dist_url", "w") as f:
                f.writelines([ip_dist_url])

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    if os.path.exists(".ip_dist_url") and rank == 0:
        os.remove(".ip_dist_url")

    torch.cuda.set_device(local_rank)
    func(cfg)
