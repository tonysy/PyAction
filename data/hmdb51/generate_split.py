import os
from collections import defaultdict

label_folder = "/public/sist/home/hexm/Datasets/HMDB51/testTrainMulti_7030_splits"
video_folder = "/public/sist/home/hexm/Datasets/HMDB51/videos"
file_list = os.listdir(label_folder)

video_items_dict = defaultdict(list)

for item in file_list:
    if "split1.txt" in item:
        cat_name = item[:-16]
        with open(os.path.join(label_folder, item), "r") as f:
            video_items_dict[cat_name].extend(f.readlines())

# assert len(video_items_list) == 7030

train_split = []
test_split = []

cat_mapping = dict()
for idx, cat_name in enumerate(sorted(video_items_dict.keys())):
    cat_mapping[idx] = cat_name

    for line in video_items_dict[cat_name]:
        name, test_flag = line.strip().split(" ")
        path = os.path.join(video_folder, cat_name, name)
        assert os.path.exists(path), "{} not exists".format(path)

        if test_flag == "1":
            train_split.append("{} {}\n".format(path, idx))
        elif test_flag == "2":
            test_split.append("{} {}\n".format(path, idx))

with open("train.csv", "w") as f:
    f.writelines(train_split)

with open("val.csv", "w") as f:
    f.writelines(test_split)

with open("test.csv", "w") as f:
    f.writelines(test_split)
