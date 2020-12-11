# on G-Clsuter of ShanghaiTech

IN_DATA_DIR="/data/datasets/video/sth_sth_v2/20bn-something-something-v2/"
OUT_DATA_DIR="/data/datasets/video/sth_sth_v2/frames_ffmpeg_4.1.4/"
ffmpeg_static="/data/softwares/ffmpeg-4.1.4-amd64-static/ffmpeg"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

res=$(ls -A1 -U ${IN_DATA_DIR} | wc -l)
j=1
for video in $(ls -A1 -U ${IN_DATA_DIR})
do
  video_name=${video##*/}
  echo -n "[$j / $res] $video_name" $'\r'
  ((j++))
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