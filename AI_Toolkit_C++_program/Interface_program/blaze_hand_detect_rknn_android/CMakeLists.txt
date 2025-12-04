cmake_minimum_required(VERSION 3.10)
project(blaze_hand_detect_project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/blaze_hand_detect_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

include_directories(${CMAKE_SOURCE_DIR})

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588 or rv110x")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty)

# opencv
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
else()
  if(LIB_ARCH STREQUAL "armhf")
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  endif()
endif()
find_package(OpenCV REQUIRED)

# rga
if(TARGET_SOC STREQUAL "rk356x")
  set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga/RK356X)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RGA_PATH ${CMAKE_SOURCE_DIR}/../3rdparty/rga/RK3588)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588")
endif()
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RGA_LIB ${RGA_PATH}/lib/Android/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
else()
  set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)
endif()
include_directories(${RGA_PATH}/include)

set(CMAKE_INSTALL_RPATH "lib")

# blaze_hand_detect_demo
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------
# ⭐⭐ 关键新增：查找 Android log 库 ⭐⭐
# ---------------------------------------------------------
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(log-lib log)
endif()

add_library(hand_detect_interface SHARED
        src/hand_detect_interface.cpp
        src/blaze_hand_detect_rknn.cpp
        src/utils.cpp
        src/event_control.cpp
        src/debug.cpp
)

# ---------------------------------------------------------
# ⭐⭐ 关键新增：链接 Android log 库 ⭐⭐
# ---------------------------------------------------------
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(hand_detect_interface
        ${RKNN_RT_LIB}
        ${RGA_LIB}
        ${OpenCV_LIBS}
        ${log-lib}   # <-- ★★★ 加入这里！
    )
else()
    target_link_libraries(hand_detect_interface
        ${RKNN_RT_LIB}
        ${RGA_LIB}
        ${OpenCV_LIBS}
    )
endif()

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/hand_detect_interface_${CMAKE_SYSTEM_NAME})
install(TARGETS hand_detect_interface DESTINATION ./)

install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)
install(DIRECTORY model DESTINATION ./)
install(DIRECTORY config DESTINATION ./)
install(FILES 
    ${CMAKE_SOURCE_DIR}/include/hand_detect_interface.hpp ./Android_JNI_Interface_usage_demo DESTINATION ./)
