#pragma once

#include <string>
#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.hpp"
#include "rga.h"
#include "types.hpp"
#include "rknn_api.h"
#include "utils.hpp"
#include <unordered_map>
#include <fstream>

namespace HandDetectRknn{



class Detector {
    public:
        Detector();
        ~Detector();

        ConfigInfo getModelparameter();

        bool loadModel(const std::string& model_path);

        std::vector<PalmBox>  infer(const std::vector<float>& input);

        
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

        std::unordered_map<std::string, std::string> readConfig(const std::string& filename,ConfigInfo &cfg_values);
        std::vector<float> loadAnchorsBin(const std::string& filename) ;

        rknn_context ctx_ = 0;
        std::vector<char> model_data_;
        rknn_input_output_num io_num_;

        const std::string config_file_ = "./config/cfg.txt"; // 模型参数文件
        std::vector<float> anchors_;                         // anchors信息
        ConfigInfo cfg_values_;
        
};

}
