#include <iostream>
#include <sys/time.h>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "event_control.hpp"

using namespace HandDetectRknn;

#define YUV420_NV21 0

int main(int argc, char** argv) {
    struct timeval start_time, stop_time;

    // 读取配置文件信息
    ConfigInfo config_info = readConfig("./config/cfg.txt");

    // 3. 创建 Detector 
    Detector detector(config_info);  
    HandDetectStateController handDetectState(config_info.max_frame_threshold);

    std::string image_path = "./model/hand.png";
    std::string model_path = config_info.model_path;
    if (!detector.loadModel(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // 计算耗时
    gettimeofday(&start_time, NULL);

    # if YUV420_NV21
        const uint8_t* nv21_input;
        int image_width;
        int image_height;
        std::vector<PalmBox>  results = detector.infer_nv21_zero_copy(nv21_input,image_width,image_height);
    # else
        // std::vector<PalmBox>  results = detector.infer_image_rga_zero_copy(image_path);
        // 1. 读取图片
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return -1;
        }
        // 2. 预处理
        std::vector<float> input_data = preprocess_image(img, config_info.input_width, config_info.input_height);
        std::vector<PalmBox>  results = detector.infer(input_data);
    
    #endif
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    

    handDetectState.update(results.empty() ? false : true);  //更新手势识别，检测状态
    
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
