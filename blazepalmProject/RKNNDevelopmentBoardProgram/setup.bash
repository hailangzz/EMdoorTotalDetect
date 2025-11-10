#!/bin/bash
# toolkit2 环境初始化脚本

# 获取当前脚本所在目录（绝对路径）
THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 激活 Conda 环境
source $THIS_DIR/toolkit2/bin/activate

# 修复路径（只需第一次执行时）
if [ ! -f "$THIS_DIR/toolkit2/.unpacked" ]; then
    conda-unpack
    touch "$THIS_DIR/toolkit2/.unpacked"
fi

echo "[toolkit2_env] Conda 环境已激活。"
