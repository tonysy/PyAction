# Extract frames from videos.
IN_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/videos_15min_static"
OUT_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/frames_static"


for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  folder_name=${OUT_DATA_DIR}/${video_name}/
  echo ${folder_name} $(ls ${folder_name} | wc -l)
done