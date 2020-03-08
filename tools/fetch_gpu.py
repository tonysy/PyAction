"""
@Description: Get GPU Information
@Author: Songyang Zhang
@Email: sy.zhangbuaa@gmail.com
@License: (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University:
@Date: 2020-03-08 01:48:01
@LastEditors: Songyang Zhang
@LastEditTime: 2020-03-08 03:13:42
"""

from gpustat.core import GPUStatCollection

gpu_stats = GPUStatCollection.new_query()
info = gpu_stats.jsonify()["gpus"]
gpu_u = []
for idx, each in enumerate(info):
    mem_ratio = each["memory.used"] / each["memory.total"]
    u_ratio = each["utilization.gpu"]
    if mem_ratio < 0.10 and u_ratio < 10:
        # if mem_ratio < 99.10 and u_ratio < 99:
        gpu_u.append(idx)

gpu_id_str = ""
for idx in range(len(gpu_u)):
    gpu_id_str += "%d" % gpu_u[idx]
    if idx != len(gpu_u) - 1:
        gpu_id_str += ","

print("%d %s" % (len(gpu_u), gpu_id_str))
