# AVA Dataset Preparation
**NOTICE! The FFmpeg version is important, different version will affect the performance significantly!**

## 1. FFmpeg Download
```
https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
```
- Ref: Solution from the [Github Issues](https://github.com/facebookresearch/video-long-term-feature-banks/issues/29)

## 2. Video Download
```
bash data/ava/tools/ava_video_download.sh
```

## 3. Video Clips Cut
```
bash data/ava/tools/cut_ava_videos.sh
```

## 4. Video Frames extract
```
bash data/ava/tools/extract_ava_frames.sh
```

## Download annotations.

