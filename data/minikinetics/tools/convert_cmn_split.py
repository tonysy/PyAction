# Please put cmn split folder, i.e. kinetics-100/[train/val/test].list in this folder
import json
import os
from tqdm import tqdm

import argparse
import random


def parse_args():
    """
    Parse the following arguments for the data processing pipeline.
    Args:

        path_prefix (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        """
    parser = argparse.ArgumentParser(
        description="Provide PyAction Kinetics Data Processing."
    )

    parser.add_argument(
        "--path-prefix",
        help="Folder of Kinetics dataset",
        default=os.path.expanduser("~/Datasets/kinetics-400/raw-part/compress"), #"/public/sist/home/hexm/Datasets/",
        type=str,
    )

    parser.add_argument(
        '--replaced',
        action='store_true', 
        default=False,
        help='Use repalced video or not'
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    return parser.parse_args()


if __name__ == "__main__":
        
    args = parse_args()

    if args.replaced:
        folder = "kinetics-100-replaced"
        fn_suffix = "_replaced"
    else:
        folder = "kinetics-100"
        fn_suffix = ""

    for name in ["train", "val", "test"]:
        f = open("{}/{}.list".format(folder, name), "r")
        assert f is not None
        lines = f.readlines()
        f.close()

        # gen label
        set_classes = set()
        for line in lines:
            c = line.split("/")[0].replace(" ", "_")
            set_classes.add(c)
        dict_class_label = {k:i for i, k in enumerate(set_classes)}
        print(dict_class_label)

        # gen csv
        with open("{}{}.csv".format(name, fn_suffix), "w") as csv_file:
            for line in lines:
                line = line.replace(" ", "_").rstrip("\n")
                c = line.split("/")[0]
                l = dict_class_label[c]
                csv_file.writelines(
                    os.path.expanduser("{}/train_256/{}.mp4 {}\n").format(args.path_prefix, line, l)
                )
        
        # verify csv
        with open("{}{}.csv".format(name, fn_suffix), "r") as csv_file:
            lines = csv_file.readlines()
            print("{}: {} lines.".format(name, len(lines)))
