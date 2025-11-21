cmake_minimum_required(VERSION 3.10)
project(blazepalm_video_cpp)
set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNXRuntime (根据你安装路径修改)
if(DEFINED ENV{ONNXRUNTIME_HOME})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_HOME})
else()
    message(FATAL_ERROR "Please set ONNXRUNTIME_HOME environment variable to ONNXRuntime root folder")
endif()

# include 路径
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
)

# 链接库路径
link_directories(${ONNXRUNTIME_ROOT}/lib)

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB SRC_FILES src/*.cpp)
# 可执行文件
add_executable(video_reader
    ${SRC_FILES}

)

# 链接库
target_link_libraries(video_reader
    ${OpenCV_LIBS}
    onnxruntime
)
