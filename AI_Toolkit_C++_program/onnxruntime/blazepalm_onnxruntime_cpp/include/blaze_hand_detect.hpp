#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "types.hpp"

namespace HandDetect{

  class Detector{

    public:
      Detector() = default;
      ~Detector();

    bool loadModel(const char* model_path);
    void printModelInfo() const;
    bool preprocess(const cv::Mat& img, std::vector<float>& input_data, std::vector<int64_t>& input_shape);

    std::vector<float> infer(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape);
    std::vector<PalmBox> parseOutput(const std::vector<float>& output_data, float score_threshold=0.7f, float iou_threshold=0.3f);
    
    std::pair<std::vector<float>, std::vector<float>> infer_output2(const std::vector<float>& input_data,const std::vector<int64_t>& input_shape);
    std::vector<PalmBox> parseOutput_output2(
        const std::vector<float>& regress,
        const std::vector<float>& scores,
        float score_threshold,
        float iou_threshold);

    void decodeBoxes(const std::vector<float>& raw_boxes,
                 const std::vector<float>& anchors, // 每个 anchor 4 float: cx,cy,w,h
                 std::vector<PalmBox>& boxes_out,
                 int num_boxes,
                 int num_keypoints,
                 float resolution);
                 
    std::vector<PalmBox> rawOutputToDetections(const std::vector<float>& raw_boxes,
                                           const std::vector<float>& raw_scores,
                                           const std::vector<float>& anchors,
                                           int num_boxes = 2016,
                                           int num_keypoints = 7,
                                           float resolution = 192,
                                           float score_threshold = 0.7f);

    std::vector<float> test_matrix_info(Detector & hand_detect);

    private:
      Ort::AllocatorWithDefaultOptions allocator_; // 放成员里，确保生命周期
      std::unique_ptr<Ort::Env> env_;
      std::unique_ptr<Ort::Session> session_;
      

  };

}