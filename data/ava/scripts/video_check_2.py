import os

path_prefix_out = "/public/sist/home/hexm/Datasets/AVA_dataset/videos/"
has_video_list = os.listdir(path_prefix_out)
has_video_list = [item.strip().split(".")[0] for item in has_video_list]


with open("ava_file_names_trainval_v2.1.txt", "r") as f:
    all_valid_names = f.readlines()

all_valid_list = [item.strip().split(".")[0] for item in all_valid_names]

all_valid_dict = {item.strip().split(".")[0]: item for item in all_valid_names}


no_down_list = list(set(all_valid_list).difference(set(has_video_list)))

joint_down_list = list(set(all_valid_list).intersection(set(has_video_list)))

need_down_list = []
for item in no_down_list:
    need_down_list.append(all_valid_dict[item])
import pdb

pdb.set_trace()

with open("ava_file_names_trainval_v2.1_need_download.txt", "w") as f:
    f.writelines(need_down_list)
# no_down_list_2 = list(set(has_video_list).difference(set(all_valid_list)))

# cout = {item:has_video_list.count(item) for item in set(has_video_list)}
# for item in set(has_video_list):
#     if cout[item] != 1:
#         print(item)
import pdb

pdb.set_trace()
