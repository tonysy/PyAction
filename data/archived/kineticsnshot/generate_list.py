import json
import os
from tqdm import tqdm

import argparse
import random

# Generate list fewshot

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

n_train_classes = 64
n_val_classes = 12
n_test_classes = 24
n_samples_per_class = 100

# dataset_path_prefix = "/public/sist/home/hexm/Datasets/"
args = parse_args()
dataset_path_prefix = args.path_prefix

data_path = os.path.join(
    dataset_path_prefix, "train_256"
)

assert os.path.exists(data_path)
all_list = os.listdir(data_path)

# ------------- Train CSV generation ----------------------

# randomly select n_train_classes classes
train_list = random.sample(all_list, n_train_classes)
train_list = sorted(train_list)
assert len(train_list) == n_train_classes #400

train_mapping = {item: idx for idx, item in enumerate(train_list)}

if not os.path.exists("./train_mapping.json"):
    with open("train_mapping.json", "w") as f:
        json.dump(train_mapping, f)

# train_list: sorted class names
# train_cat_dict: mapping class name => video list of this class(in this folder)
# train_csv_list: list of "video_path label" strings  # e.g. /root/Datasets/kinetics-400/raw-part/compress/train_256/abseiling/zxOmGsxBHGU_000185_000195.mp4 0
train_cat_dict = {}
for item in tqdm(train_list):
    # randomly select n_samples_per_class samples per class
    lst = os.listdir(os.path.join(data_path, item))
    lst = random.sample(lst, min(len(lst), n_samples_per_class))
    train_cat_dict[item] = lst

train_csv_list = []
for cat_name in tqdm(train_list):
    for video_name in train_cat_dict[cat_name]:
        path = os.path.join(data_path, cat_name, video_name)
        assert os.path.exists(path)
        label_id = train_mapping[cat_name]
        str_ = "{} {}\n".format(path, label_id)
        train_csv_list.append(str_)

with open("train.csv", "w") as f:
    f.writelines(train_csv_list)

print("Total Train {} video clips".format(len(train_csv_list)))

# ------------- Val CSV generation ----------------------

val_list = [c for c in all_list if c not in train_list]
assert not set(val_list).intersection(set(train_list))

# randomly select n_val_classes classes
val_list = random.sample(val_list, n_val_classes)
val_list = sorted(val_list)
assert len(val_list) == n_val_classes

val_mapping = {item: idx for idx, item in enumerate(val_list)}

if not os.path.exists("./val_mapping.json"):
    with open("val_mapping.json", "w") as f:
        json.dump(val_mapping, f)

val_cat_dict = {}
for item in tqdm(val_list):
    # randomly select n_samples_per_class samples per class
    lst = os.listdir(os.path.join(data_path, item))
    lst = random.sample(lst, min(len(lst), n_samples_per_class))
    val_cat_dict[item] = lst

val_csv_list = []
for cat_name in tqdm(val_list):
    for video_name in val_cat_dict[cat_name]:
        path = os.path.join(data_path, cat_name, video_name)
        assert os.path.exists(path)
        label_id = val_mapping[cat_name]
        str_ = "{} {}\n".format(path, label_id)
        val_csv_list.append(str_)

with open("val.csv", "w") as f:
    f.writelines(val_csv_list)

print("Total Val {} video clips".format(len(val_csv_list)))


# ------------- Test CSV generation ----------------------
used_list = train_list + val_list
test_list = [c for c in all_list if c not in used_list]
assert not set(test_list).intersection(set(used_list))

# randomly select n_test_classes classes
test_list = random.sample(test_list, n_test_classes)
test_list = sorted(test_list)
assert len(test_list) == n_test_classes

test_mapping = {item: idx for idx, item in enumerate(test_list)}

if not os.path.exists("./test_mapping.json"):
    with open("test_mapping.json", "w") as f:
        json.dump(test_mapping, f)

test_cat_dict = {}
for item in tqdm(test_list):
    # randomly select n_samples_per_class samples per class
    lst = os.listdir(os.path.join(data_path, item))
    lst = random.sample(lst, min(len(lst), n_samples_per_class))
    test_cat_dict[item] = lst

test_csv_list = []
for cat_name in tqdm(test_list):
    for video_name in test_cat_dict[cat_name]:
        path = os.path.join(data_path, cat_name, video_name)
        assert os.path.exists(path)
        label_id = test_mapping[cat_name]
        str_ = "{} {}\n".format(path, label_id)
        test_csv_list.append(str_)

with open("test.csv", "w") as f:
    f.writelines(test_csv_list)

print("Total test {} video clips".format(len(test_csv_list)))



