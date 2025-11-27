#ifndef DETECTOR_H
#define DETECTOR_H

#include <stdint.h>  // for uint8_t

#ifdef __cplusplus
extern "C" {
#endif

struct AndroidImageNV21 {
    uint8_t* image_input_nv21;   // NV21 数据
    int image_width;
    int image_height;
};

// 目标检测接口（使用指针，更兼容 C/JNI）
bool hand_detect_interface(AndroidImageNV21* image_object_input);

#ifdef __cplusplus
}
#endif

#endif // DETECTOR_H
