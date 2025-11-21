#include <string>
#include <opencv2/opencv.hpp>
#include "types.hpp"
#include <iostream>


bool GetVideoInfo(const std::string& video_path, int& width, int& height, double& fps, int& frame_count);
bool resizeImage(const std::string& image_path,std::vector<float>& output_data,std::vector<int64_t>& input_shape);


static float computeIOU(const PalmBox& a, const PalmBox& b);
std::vector<PalmBox> nms(const std::vector<PalmBox>& boxes, float iou_threshold);
float sigmoid(float x);
std::vector<float> loadAnchorsBin(const std::string& filename);
void drawPalmBoxes(cv::Mat& image, const std::vector<PalmBox>& boxes);







