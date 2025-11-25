#include <iostream>
#include <opencv2/opencv.hpp>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "types.hpp"

using namespace HandDetectRknn;

// 将 cv::Mat 转为 float 数组，HWC，范围 [0,1]


int main(int argc, char** argv) {
 
    Detector detector;
    ConfigInfo model_config_info = detector.getModelparameter();

    // std::string model_path = "./model/palm_detection_full.rknn";
    std::string model_path = model_config_info.model_path;
    std::string image_path = "./model/hand.png";

    // 1. 读取图片
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }
    // 2. 预处理
    std::vector<float> input_data = preprocess_image(img, model_config_info.input_width, model_config_info.input_height);

    // 3. 创建 Detector
    
    if (!detector.loadModel(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    std::vector<PalmBox>  results = detector.infer(input_data);

    // 5. 打印结果
    for (size_t i = 0; i < results.size(); ++i) {
        const PalmBox& det = results[i];
        std::cout << "Detection " << i
                  << " score=" << det.score
                  << " bbox=(" << det.x << "," << det.y << "," 
                  << det.w << "," << det.h << ")"
                  << " keypoints=[";
        for (auto kp : det.keypoints) std::cout << kp << " ";
        std::cout << "]" << std::endl;
    }

    plotDetectBoxs(image_path,results);

    return 0;
}
