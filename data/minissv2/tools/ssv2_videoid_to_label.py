import os
import json

PATH_TO_DATA_DIR = "../ssv2/"

with open(
    os.path.join(
        PATH_TO_DATA_DIR,
        "labels/something-something-v2-labels.json",
    ),
    "r",
) as f:
    label_dict = json.load(f)


video_names = []
labels = []


train_label_file = os.path.join(
    PATH_TO_DATA_DIR,
    "labels/something-something-v2-{}.json".format("train"),
)

with open(train_label_file, "r") as f:
    train_label_json = json.load(f)


for video in train_label_json:
    video_name = video["id"]
    template = video["template"]
    template = template.replace("[", "")
    template = template.replace("]", "")
    label = int(label_dict[template])
    video_names.append(video_name)
    labels.append(label)


val_label_file = os.path.join(
    PATH_TO_DATA_DIR,
    "labels/something-something-v2-{}.json".format("validation"),
)

with open(val_label_file, "r") as f:
    val_label_json = json.load(f)

for video in val_label_json:
    video_name = video["id"]
    template = video["template"]
    template = template.replace("[", "")
    template = template.replace("]", "")
    label = int(label_dict[template])
    video_names.append(video_name)
    labels.append(label)

all_label_dict = dict(zip(video_names, labels))

with open("video_id_to_label.json", "w") as f:
    f.write(json.dumps(all_label_dict))
    f.flush()
