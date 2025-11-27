#pragma once

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
    std::vector<float> keypoints;
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
    int max_frame_threshold;

};

struct AndroidImageNV21 {
    uint8_t* image_input_nv21;   // 注意去掉 const，因为你要分配内存
    int image_width;
    int image_height;
};