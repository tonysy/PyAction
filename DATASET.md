# Dataset 
## Kinetics
### Get Data
- (Option-1) Process the RAW videos

The Kinetics Dataset could be downloaded via the code released by ActivityNet:

1. Download the videos via the official [scripts](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).

2. After all the videos were downloaded, resize the video to the short edge size of 256, then prepare the csv files for training, validation, and testing set as `train.csv`, `val.csv`, `test.csv`. The format of the csv file is:

```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

All the Kinetics models in the Model Zoo are trained and tested with the same data as [Non-local Network](https://github.com/facebookresearch/video-nonlocal-net/blob/master/DATASET.md). For dataset specific issues, please reach out to the [dataset provider](https://deepmind.com/research/open-source/kinetics).

- (Option-2) Download Xiaolong's Data

Download data from: [Dropbox Link](https://www.dropbox.com/s/wcs01mlqdgtq4gn/compress.tar.gz).

### Generate Video List
Enter the data/kinetics to generate the data split list by:
```bash
# You need to edit `dataset_path_prefix` in the generate_list.py to you own path
cd ./data/kinetics
python generate_list.py
```
Then you will get `train.csv`, `val.csv` and `test.csv`.
