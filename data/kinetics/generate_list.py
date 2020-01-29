import json
import os
from tqdm import tqdm

data_path = (
    "/public/sist/home/hexm/Datasets/kinetics-400/" "raw-part/compress/train_256"
)

assert os.path.exists(data_path)
cat_list = os.listdir(data_path)
assert len(cat_list) == 400

cat_mapping = {item: idx for idx, item in enumerate(cat_list)}
with open("cat_mapping.json", "w") as f:
    json.dump(cat_mapping, f)

# ------------- Train CSV generation ----------------------
train_data_path = (
    "/public/sist/home/hexm/Datasets/kinetics-400/" "raw-part/compress/train_256"
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

val_data_path = (
    "/public/sist/home/hexm/Datasets/kinetics-400/" "raw-part/compress/val_256"
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

print("Total Val {} video clips".format(len(val_csv_list)))
