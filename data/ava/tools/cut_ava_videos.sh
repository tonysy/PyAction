# Cut each video from its 15th to 30th minute.

IN_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/videos"
OUT_DATA_DIR="/public/sist/home/hexm/Datasets/ava_new/videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    ffmpeg -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done