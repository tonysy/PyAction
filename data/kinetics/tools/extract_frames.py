import os
import av
from tqdm import tqdm
import json


def extract_frames(folder, out_folder, mode="train_256"):
    split_folder = os.path.join(folder, mode)
    assert os.path.exists(split_folder), split_folder

    video_dicts = dict()
    cat_list = os.listdir(split_folder)
    assert len(cat_list) == 400, len(cat_list)
    video_metas = []

    for cat in tqdm(cat_list):
        cat_folder = os.path.join(split_folder, cat)
        video_dicts[cat] = os.listdir(cat_folder)

        for one_video in video_dicts[cat]:
            video_path = os.path.join(split_folder, cat, one_video)
            assert os.path.exists(video_path), video_path
            container = av.open(video_path)

            # get video meta
            fps = float(container.streams.video[0].average_rate)
            frames_length = container.streams.video[0].frames
            duration = container.streams.video[0].duration

            # extrac frames
            out_frame_folder = os.path.join(
                out_folder, mode, cat, one_video.split(".")[0]
            )
            if os.path.exists(out_frame_folder):
                print("{} Exists!".format(out_frame_folder))
                import pdb

                pdb.set_trace()
                continue
            else:
                os.makedirs(out_frame_folder)

            video_metas.append(
                {
                    "fps": fps,
                    "frames_length": frames_length,
                    "duration": duration,
                    "video_path": video_path,
                    "mode": mode,
                    "category": cat,
                    "video_name": one_video,
                    "frames_folder": out_frame_folder,
                }
            )
            for frame in container.decode(video=0):
                frame.to_image().save(
                    "{}/frames_{}.jpg".format(out_frame_folder, frame.pts)
                )

    return video_metas


folder = "/public/sist/home/hexm/Datasets/kinetics-400/raw-part/compress"
out_folder = "/public/sist/home/hexm/Datasets/kinetics-400/frames"
video_metas = extract_frames(folder, out_folder)


with open("train_video_meta.json", "w") as f:
    f.write(json.dumps({"meta": video_metas}))
    f.flush()
