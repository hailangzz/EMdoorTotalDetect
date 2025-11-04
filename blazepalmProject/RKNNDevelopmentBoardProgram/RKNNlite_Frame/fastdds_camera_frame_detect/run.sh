#!/bin/bash
source /home/robot/miniforge3/bin/activate toolkit2
source /home/robot/zhangzhuo/fastdds_env/setup.bash
export ROSCONSOLE_CONFIG_FILE=./roslogging.conf


python3 hand_detect.py
