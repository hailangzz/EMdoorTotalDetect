#!/bin/bash
source /home/robot/miniforge3/bin/activate toolkit2
source /home/robot/zhangzhuo/fastdds_env/setup.bash

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
export OMP_NUM_THREADS=1

python3 hand_detect.py
