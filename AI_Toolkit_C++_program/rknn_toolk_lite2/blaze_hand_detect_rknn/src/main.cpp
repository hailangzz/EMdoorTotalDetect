#include <iostream>
#include <opencv2/opencv.hpp>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "types.hpp"

using namespace HandDetectRknn;

// 将 cv::Mat 转为 float 数组，HWC，范围 [0,1]
std::vector<float> preprocess(const cv::Mat& img, int target_w, int target_h) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));
    cv::Mat img_float;
    resized.convertTo(img_float, CV_32FC3, 1.0 / 255.0); // 归一化

    std::vector<float> input_data(target_w * target_h * 3);
    int idx = 0;
    for (int h = 0; h < target_h; ++h) {
        for (int w = 0; w < target_w; ++w) {
            cv::Vec3f pixel = img_float.at<cv::Vec3f>(h, w);
            input_data[idx++] = pixel[2]; // R
            input_data[idx++] = pixel[1]; // G
            input_data[idx++] = pixel[0]; // B
        }
    }
    return input_data;
}

int main(int argc, char** argv) {
 

    std::string model_path = "./model/palm_detection_full.rknn";
    std::string image_path = "./model/hand.png";

    // 1. 读取图片
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    // 假设模型输入尺寸为 224x224
    int target_w = 192;
    int target_h = 192;

    // 2. 预处理
    std::vector<float> input_data = preprocess(img, target_w, target_h);

    // 3. 创建 Detector
    Detector detector;
    if (!detector.loadModel(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // 4. 推理
    std::vector<int64_t> shape = {1, target_h, target_w, 3}; // HWC
    std::vector<PalmBox>  results = detector.infer(input_data, shape);

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
