import os

path_prefix = "/public/sist/home/hexm/Datasets/AVA/"
train_video_list = os.listdir(os.path.join(path_prefix, "train"))
test_video_list = os.listdir(os.path.join(path_prefix, "test"))

with open("ava_file_names_trainval_v2.1.txt", "r") as f:
    all_list = f.readlines()

all_list_name = [item.strip().split(".")[0] for item in all_list]

no_list = list(set(all_list_name).difference(set(train_video_list + test_video_list)))


need_download_list = []
for item in all_list:
    if item.strip().split(".")[0] in no_list:
        need_download_list.append(item)

with open("ava_file_names_trainval_v2.1_need_download.txt", "w") as f:
    f.writelines(need_download_list)

import pdb

pdb.set_trace()
