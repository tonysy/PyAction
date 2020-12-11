import os

data_folder = "/data/datasets/video/sth_sth_v2/20bn-something-something-v2/"
# data = glob.glob("/data/datasets/video/sth_sth_v2/20bn-something-something-v2/")

# data_list = open('cmn_split/smsm-100/test.list', 'r').readlines()
# data_list = open('cmn_split/smsm-100/train.list', 'r').readlines()
data_list = open("cmn_split/smsm-100/val.list", "r").readlines()

for item in data_list:
    index = item.strip().split("/")[-1]
    video_path = os.path.join(data_folder, index + ".webm")
    # import pdb; pdb.set_trace()
    assert os.path.exists(video_path)
    # import pdb; pdb.set_trace()
