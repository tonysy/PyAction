# Extract frames from videos.

IN_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/videos_15min_static"
OUT_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/frames_static"
ffmpeg_static="/public/sist/home/hexm/Softwares/ffmpeg-4.2.2-amd64-static/ffmpeg"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(cat $1)
do
  video_name=${video##*/}

  if [[ $video_name = *".webm" ]]; then
    video_name=${video_name::-5}
  else
    video_name=${video_name::-4}
  fi

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  rm -rf "${out_video_dir}"
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  $ffmpeg_static -i "${IN_DATA_DIR}/${video}" -r 30 -q:v 1 "${out_name}"
done