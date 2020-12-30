import os
from tqdm import tqdm
from collections import defaultdict
import json
import argparse


def get_data_list(src_data_list, data_folder, image_paths):
    data_list = dict()

    for item in src_data_list:
        # index = item.strip().split("/")[-1]
        index = item.strip().split(" ")[0]
        video_path = os.path.join(data_folder, index + ".webm")
        assert os.path.exists(video_path)

        data_list[index] = {
            "frames": image_paths[index],
            "video_path": video_path,
        }
    return data_list


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
        "--video-folder",
        help="Folder of Kinetics dataset",
        # default=os.path.expanduser("./"), #"/public/sist/home/hexm/Datasets/",
        type=str,
    )

    parser.add_argument(
        "--frames-folder",
        help="Folder of Kinetics dataset",
        # default=os.path.expanduser(""), #"/public/sist/home/hexm/Datasets/",
        type=str,
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # video_folder = "/data/datasets/video/sth_sth_v2/20bn-something-something-v2/"
    # frames_folder = "/data/datasets/video/sth_sth_v2/frames_ffmpeg_4.1.4"
    video_folder = args.video_folder
    frames_folder = args.frames_folder

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

    src_data_list = open("otam_split/somethingv2_meta_train_train.txt", "r").readlines()
    src_data_list += open("otam_split/somethingv2_meta_train_val.txt", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_train_large.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    src_data_list = open("otam_split/somethingv2_meta_test.txt", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_test_large.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    src_data_list = open("otam_split/somethingv2_meta_val.txt", "r").readlines()
    dst_data_list = get_data_list(src_data_list, video_folder, image_paths)
    with open("fewshot_val_large.json", "w") as f:
        f.write(json.dumps(dst_data_list))
        f.flush()

    # import pdb; pdb.set_trace()
