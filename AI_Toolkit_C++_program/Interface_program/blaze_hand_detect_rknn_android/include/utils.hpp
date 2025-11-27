#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <im2d.h>
#include <rga.h>


ConfigInfo readConfig(const std::string& filename);

AndroidImageNV21 load_image_to_nv21(const std::string& image_path);

bool nv21_to_rgb_resize(const uint8_t* nv21_data,int src_w, int src_h,uint8_t* rgb_data,int dst_w, int dst_h);

void rgb_to_float(const uint8_t* rgb, float* out, int w, int h);

bool resizeImage(const std::string& image_path,std::vector<float>& output_data,std::vector<int64_t>& input_shape);

std::vector<float> preprocess_image(const cv::Mat& img, int target_w, int target_h);

static float computeIOU(const PalmBox& a, const PalmBox& b);

std::vector<PalmBox> nms(const std::vector<PalmBox>& boxes, float iou_threshold);

float sigmoid(float x);

std::vector<float> loadAnchorsBin(const std::string& filename);

void drawPalmBoxes(cv::Mat& image, const std::vector<PalmBox>& boxes);

void plotDetectBoxs(const std::string & image_path,const std::vector<PalmBox>& boxes);

std::unordered_map<std::string, std::string> readConfig(const std::string& filename,ConfigInfo &cfg_values);   // 读取配置文件信息

double __get_us(struct timeval t);






