set -e

TARGET_SOC="rk3588"
export GCC_COMPILER=/home/chenkejing/RKNNProjects/CrossCompileToolChain/Linux/aarch64/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu

export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}

cmake ../.. \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DTARGET_SOC=${TARGET_SOC} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

make -j4
make install

cd -
