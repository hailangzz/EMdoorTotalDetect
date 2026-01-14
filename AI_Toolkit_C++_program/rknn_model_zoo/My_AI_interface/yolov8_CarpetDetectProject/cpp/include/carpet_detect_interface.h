// yolov8_api.h
#pragma once
#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif


// 定义单个 3D 坐标结构体
typedef struct {
    float X;
    float Y;
    float Z;
} CameraCoordinate;

// 定义检测结果结构体
typedef struct {
    float prop;                     // 置信度
    int cls_id;                      // 类别 ID
    CameraCoordinate coords[4];      // 最多存储 4 个 3D 坐标点                // 实际存储的坐标点数量
} ObjectCameraDetectResult;


bool carpet_model_init(const char* config_path);
bool carpet_detect_infer(const cv::Mat& img,ObjectCameraDetectResult* results, int max_results=64);
void carpet_model_release();

#ifdef __cplusplus
}
#endif
