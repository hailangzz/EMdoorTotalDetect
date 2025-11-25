#pragma once
#include <stdio.h>
#include <stdlib.h>

struct VideoInfo{
  std::string video_path = "test.mp4";
  int width = 0;
  int height = 0;
  double fps = 0.0;
  int frame_count = 0;
};


struct PalmBox {
    float score;
    float x, y, w, h;
    std::vector<float> keypoints; // 14 float, x1,y1,...,x7,y7
};

struct ConfigInfo {
    // const char* model_path;
    std::string model_path;
    int input_width;
    int input_height;
    std::string anchors_path;
    int num_boxes;
    int num_keypoints; 
    float resolution;
    float score_threshold;

};

struct Detection {
  float score;
  float x, y, w, h;
  std::vector<float> keypoints;
};
