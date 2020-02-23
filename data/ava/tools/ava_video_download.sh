DATA_DIR="//public/sist/home/hexm/Datasets/ava_new/videos"

for line in $(cat ava_file_names_trainval_v2.1.txt)
do
  wget -c https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done