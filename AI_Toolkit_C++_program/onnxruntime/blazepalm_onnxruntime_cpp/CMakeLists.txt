cmake_minimum_required(VERSION 3.10)
project(blazepalm_onnxruntime_cpp)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
find_package(OpenCV REQUIRED)

# ROS（如果你需要订阅图片）
find_package(catkin REQUIRED COMPONENTS
    roscpp
    image_transport
    cv_bridge
    sensor_msgs
)

catkin_package()

# ONNXRuntime
if(DEFINED ENV{ONNXRUNTIME_HOME})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_HOME})
else()
    message(FATAL_ERROR "Please set ONNXRUNTIME_HOME")
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_ROOT}/include
    ${catkin_INCLUDE_DIRS}
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

file(GLOB SRC_FILES src/*.cpp)

add_executable(video_reader ${SRC_FILES})

target_link_libraries(video_reader
    ${OpenCV_LIBS}
    onnxruntime
    ${catkin_LIBRARIES}
)

# 可执行文件输出位置（推荐）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/devel/lib/${PROJECT_NAME})
