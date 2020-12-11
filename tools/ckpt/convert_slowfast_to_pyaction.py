import argparse

# import os
# import torch
import pickle


def parse_args():
    """
    Parse the following arguments for checkpoint convert.
    Args:

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-ckpt",
        help="The file path of the original checkpoint",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    origin_ckpt = pickle.load(open(args.src_ckpt, "rb"), encoding="latin1")
    import pdb

    pdb.set_trace()
