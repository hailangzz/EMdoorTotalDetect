#pragma once
#include <vector>
#include <string>
#include "types.hpp"
#include "rknn_api.h"
#include "utils.hpp"
#include <unordered_map>
#include <fstream>

namespace HandDetectRknn{



class Detector {
    public:
        Detector(const ConfigInfo& config);
        ~Detector();

        void initModelparameter(ConfigInfo config_info);

        bool loadModel(const std::string& model_path);

        //opencv 读取图像
        std::vector<PalmBox> infer(const std::vector<float>& input);
        std::vector<PalmBox> infer_image_rga_zero_copy(const std::string& image_path);
        
         // 直接摄像头 NV21 数据
        std::vector<PalmBox> infer_nv21(const uint8_t* nv21_input, int src_w, int src_h);
        // 摄像头 NV21 数据  ，零拷贝                          
        std::vector<PalmBox> infer_nv21_zero_copy(const uint8_t* nv21_input,int src_w, int src_h);
        std::vector<PalmBox> infer_nv21_zero_copy_BGR(const uint8_t* nv21_input, int src_w, int src_h);
        
        std::vector<PalmBox> parseRknnOutputs(
                                    const std::vector<rknn_output>& outputs,
                                    const std::vector<float>& anchors,
                                    int num_boxes,
                                    int num_keypoints,
                                    float resolution,
                                    float score_threshold);
        

    private:

        void decodeBoxes(const std::vector<float>& raw_boxes,
                 const std::vector<float>& anchors,
                 std::vector<PalmBox>& boxes_out,
                 int num_boxes,
                 int num_keypoints,
                 float resolution);

        std::vector<PalmBox> rawOutputToDetections(const std::vector<float>& raw_boxes,
                                           const std::vector<float>& raw_scores,
                                           const std::vector<float>& anchors,
                                           int num_boxes,
                                           int num_keypoints,
                                           float resolution,
                                           float score_threshold);

        std::vector<float> loadAnchorsBin(const std::string& filename) ;

        rknn_context ctx_ = 0;
        std::vector<char> model_data_;
        rknn_input_output_num io_num_;

        const std::string config_file_ = "./config/cfg.txt"; // 模型参数文件
        std::vector<float> anchors_;                         // anchors信息
        ConfigInfo cfg_values_;
        
};

}
