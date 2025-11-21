#!/bin/bash

# -------------------------
# 脚本说明
# -------------------------
# 自动构建 blazepalm_video_cpp 项目
# 生成 compile_commands.json
# -------------------------

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")
BUILD_DIR="$PROJECT_ROOT/build"
CLEAN_BUILD=0  # 设置为1可清理旧 build

# 清理旧 build
if [ $CLEAN_BUILD -eq 1 ]; then
    echo "Cleaning old build directory..."
    rm -rf "$BUILD_DIR"
fi

# 创建 build 目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

# 运行 CMake 配置
echo "Running CMake..."
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译
echo "Building project..."
make -j$(nproc)

# 完成提示
if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Executable located in: $BUILD_DIR"
else
    echo "Build failed!"
fi
