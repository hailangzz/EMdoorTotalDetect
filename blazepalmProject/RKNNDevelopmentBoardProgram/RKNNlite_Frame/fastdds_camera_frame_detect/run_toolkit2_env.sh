#!/bin/bash
# 确保出错时脚本立即退出
set -e

# 切换工作目录
cd /home/robot/zhangzhuo/hand_detect

# 记录启动时间
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting hand detect..." >> ./run.log

# 激活conda环境
#source /home/robot/miniforge3/bin/activate toolkit2
source /home/robot/zhangzhuo/toolkit2_env/setup.bash
# 加载fastdds环境变量
source /home/robot/zhangzhuo/fastdds_env/setup.bash

# 启动python脚本，并将日志写入文件
python3 main_hand_detect.py >> ./run.log 2>&1
