
# Mini Sth-sth v2 dataset
## Create label mapping dict
```
python tools/ssv2_videoid_to_label.py
```
## For mini-ssv2-small
```
python tools/convert_cmn_split.py \
    --video-folder \
    --frames-folder \
```

## For mini-ssv2-large
Example:

```
python tools/convert_otam_split.py \
    --video-folder /data/datasets/video/sth_sth_v2/20bn-something-something-v2 \
    --frames-folder /data/datasets/video/sth_sth_v2/frames_ffmpeg_4.1.4
```