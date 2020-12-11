import os
from tqdm import tqdm
from collections import defaultdict
import json


def get_data_list(src_data_list, data_folder, image_paths):
    data_list = dict()

    for item in src_data_list:
        index = item.strip().split("/")[-1]
        video_path = os.path.join(data_folder, index + ".webm")
        assert os.path.exists(video_path)

        data_list[index] = {
            "frames": image_paths[index],
            "video_path": video_path,
        }
    return data_list


if __name__ == "__main__":

    video_folder = "/data/datasets/video/sth_sth_v2/20bn-something-something-v2/"
    frames_folder = "/data/datasets/video/sth_sth_v2/frames_ffmpeg_4.1.4"
    image_paths = defaultdict(list)
    labels = defaultdict(list)

    frame_list_file = "../ssv2/labels/train.csv"
    with open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in tqdm(f):
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if frames_folder == "":
                path = row[3]
            else:
                path = os.path.join(frames_folder, row[3])
            # import pdb; pdb.set_trace()
            assert os.path.exists(path)
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append([int(x) for x in frame_labels.split(",")])
            else:
                labels[video_name].append([])

    frame_list_file = "../ssv2/labels/val.csv"
    with open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in tqdm(f):
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if frames_folder == "":
                path = row[3]
            else:
                path = os.path.join(frames_folder, row[3])
            # import pdb; pdb.set_trace()
            assert os.path.exists(path)
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append([int(x) for x in frame_labels.split(",")])
            else:
                labels[video_name].append([])

    src_data_list = open("cmn_split/smsm-100/train.list", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_train.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    src_data_list = open("cmn_split/smsm-100/test.list", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_test.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    src_data_list = open("cmn_split/smsm-100/val.list", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_val.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    # import pdb; pdb.set_trace()
