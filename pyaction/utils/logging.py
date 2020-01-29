#!/usr/bin/env python3
# Copyright (c) ShanghaiTech PLUS Group. All Rights Reserved.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Logging."""
import builtins
import decimal
import logging
import os
import sys
import simplejson
from termcolor import colored

import pyaction.utils.distributed as du
from pyaction.utils.file_io import PathManager


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


def setup_logging(output=None, color=True, name="pyaction", abbrev_name=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    # _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "pa" if name == "pyaction" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S",
    )

    if du.is_master_proc():
        # Enable logging for the master process.
        # logging.root.handlers = []
        # logging.basicConfig(
        #     level=logging.INFO, format=_FORMAT, stream=sys.stdout
        # )

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # else:
    #     # Suppress logging for non-master processes.
    #     _suppress_print()

    # file logging: all workers
    distributed_rank = du.get_rank()
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.6f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))


def _cached_log_stream(filename):
    return PathManager.open(filename, "a")
