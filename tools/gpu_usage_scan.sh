###
 # @Description: Fetch Valid GPU Number and IDs
 # @Author: Songyang Zhang
 # @Email: sy.zhangbuaa@gmail.com
 # @License: (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University:
 # @Date: 2020-03-08 01:58:10
 # @LastEditors: Songyang Zhang
 # @LastEditTime: 2020-03-08 03:26:24
 ###
require_gpu_num=$1
target_gpu_num=$2

num_gpu=""
gpu_id=""

info="`python ${PYACTION_HOME}/tools/fetch_gpu.py`"
arr=($info)

num_gpu=${arr[0]}
gpu_id=${arr[1]}

while [ $((num_gpu)) -lt $((require_gpu_num)) ]
do
    info="`python ${PYACTION_HOME}/tools/fetch_gpu.py`"
    arr=($info)
    num_gpu=${arr[0]}
    gpu_id=${arr[1]}
    sleep 10
    time=$(date)
    # nvidia-smi
    # echo "${time} Try Once More"
done

if [ $((num_gpu)) -lt $((target_gpu_num)) ]
then
    target_gpu_num=$num_gpu
fi

echo "$target_gpu_num $gpu_id"