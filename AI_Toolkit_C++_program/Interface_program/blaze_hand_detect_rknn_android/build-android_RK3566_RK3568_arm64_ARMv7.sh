#!/bin/bash
set -e

ANDROID_NDK_PATH=/home/chenkejing/RKNNProjects/CrossCompileToolChain/Android/android-ndk-r19c
if [ -z ${ANDROID_NDK_PATH} ]; then
  ANDROID_NDK_PATH=~/opt/android-ndk-r16b
fi

BUILD_TYPE=Release
TARGET_SOC="rk356x"
ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# 支持的 ABI 列表
ABIS=("armeabi-v7a" "arm64-v8a")

for ABI in "${ABIS[@]}"; do
    BUILD_DIR=${ROOT_PWD}/build/build_android_${ABI}
    
    if [[ ! -d "${BUILD_DIR}" ]]; then
        mkdir -p ${BUILD_DIR}
    fi

    echo "=== Building for ${ABI} ==="
    cd ${BUILD_DIR}

    cmake ../.. \
        -DANDROID_TOOLCHAIN=clang \
        -DTARGET_SOC=${TARGET_SOC} \
        -DCMAKE_SYSTEM_NAME=Android \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_PATH/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI="${ABI}" \
        -DANDROID_STL=c++_static \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE}

    make -j4
    make install
    cd ..
done

echo "=== Build finished for all ABIs ==="
