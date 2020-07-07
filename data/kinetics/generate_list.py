import json
import os
from tqdm import tqdm

import argparse


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

    # if len(sys.argv) == 1:
    #     parser.print_help()
    return parser.parse_args()

# dataset_path_prefix = "/public/sist/home/hexm/Datasets/"
args = parse_args()
dataset_path_prefix = args.path_prefix

data_path = os.path.join(
    dataset_path_prefix, "train_256"
)

assert os.path.exists(data_path)
cat_list = os.listdir(data_path)
cat_list = sorted(cat_list)
assert len(cat_list) == 400

cat_mapping = {item: idx for idx, item in enumerate(cat_list)}

if not os.path.exists("./cat_mapping.json"):
    with open("cat_mapping.json", "w") as f:
        json.dump(cat_mapping, f)

# ------------- Train CSV generation ----------------------
train_data_path = os.path.join(
    dataset_path_prefix, "train_256"
)

train_cat_dict = {}
for item in tqdm(cat_list):
    train_cat_dict[item] = os.listdir(os.path.join(train_data_path, item))

train_csv_list = []
for cat_name in tqdm(cat_list):
    for video_name in train_cat_dict[cat_name]:
        path = os.path.join(train_data_path, cat_name, video_name)
        assert os.path.exists(path)
        label_id = cat_mapping[cat_name]
        str_ = "{} {}\n".format(path, label_id)
        train_csv_list.append(str_)

with open("train.csv", "w") as f:
    f.writelines(train_csv_list)

print("Total Train {} video clips".format(len(train_csv_list)))

# ------------- Val CSV generation ----------------------

val_data_path = os.path.join(
    dataset_path_prefix, "val_256"
)

val_cat_dict = {}
for item in tqdm(cat_list):
    val_cat_dict[item] = os.listdir(os.path.join(val_data_path, item))

val_csv_list = []
for cat_name in tqdm(cat_list):
    for video_name in val_cat_dict[cat_name]:
        path = os.path.join(val_data_path, cat_name, video_name)
        assert os.path.exists(path)
        label_id = cat_mapping[cat_name]
        str_ = "{} {}\n".format(path, label_id)
        val_csv_list.append(str_)

with open("val.csv", "w") as f:
    f.writelines(val_csv_list)

with open("test.csv", "w") as f:
    f.writelines(val_csv_list)

print("Total Val {} video clips".format(len(val_csv_list)))
