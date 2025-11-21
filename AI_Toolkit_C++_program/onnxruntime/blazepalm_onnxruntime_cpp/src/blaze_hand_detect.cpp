#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cassert>
#include <iomanip>
#include"blaze_hand_detect.hpp"
#include"utils.hpp"

namespace HandDetect{

bool Detector::loadModel(const char* model_path){
  
  try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx_model_loader");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(*env_, model_path, session_options);

        std::cout << "Model loaded successfully: " << model_path << "\n";
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }

  

  std::cout << "\nONNX model loaded successfully!\n";

}

void Detector::printModelInfo() const {
    if (!session_) {
        std::cerr << "Session not initialized!" << std::endl;
        return;
    }

    // 输入信息
    auto input_names = session_->GetInputNames();
    std::cout << "Number of inputs: " << input_names.size() << std::endl;

    for (size_t i = 0; i < input_names.size(); i++) {
        std::cout << "Input " << i << ": " << input_names[i] << std::endl;

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = tensor_info.GetShape();

        std::cout << "  Shape: ";
        for (auto d : input_dims) std::cout << d << " ";
        std::cout << std::endl;
    }

    // 输出信息
    auto output_names = session_->GetOutputNames();
    std::cout << "\nNumber of outputs: " << output_names.size() << std::endl;

    for (size_t i = 0; i < output_names.size(); i++) {
        std::cout << "Output " << i << ": " << output_names[i] << std::endl;

        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = tensor_info.GetShape();

        std::cout << "  Shape: ";
        for (auto d : output_dims) std::cout << d << " ";
        std::cout << std::endl;
    }
}

bool preprocess(const cv::Mat& img, std::vector<float>& input_data, std::vector<int64_t>& input_shape) {
        if (img.empty()) return false;
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(192,192));
        cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
        resized_img.convertTo(resized_img, CV_32FC3, 1.0/255.0);

        input_shape = {1,192,192,3};
        input_data.resize(1*192*192*3);

        size_t idx = 0;
        for (int h=0; h<192; h++) {
            for (int w=0; w<192; w++) {
                cv::Vec3f px = resized_img.at<cv::Vec3f>(h,w);
                for (int c=0; c<3; c++) input_data[idx++] = px[c];
            }
        }
        return true;
    }
    
std::vector<float> Detector::infer(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape) {
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    // 获取输入输出名字
    auto input_names = session_->GetInputNames();
    auto output_names = session_->GetOutputNames();

    if (input_names.empty() || output_names.empty()) {
        throw std::runtime_error("Model must have at least one input and one output");
    }
    const char* input_name = input_names[0].c_str();
    const char* output_name = output_names[0].c_str();

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

    // 推理
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        &input_name, &input_tensor, 1,
                                        &output_name, 1);

    // 获取输出张量
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    // 获取输出张量大小
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_size = type_info.GetElementCount();

    // 拷贝到 vector 返回
    std::vector<float> result(output_data, output_data + output_size);
    return result;
}

std::vector<PalmBox> Detector::parseOutput(const std::vector<float>& output_data, float score_threshold, float iou_threshold) {

  size_t num_boxes = output_data.size()/18;
  std::vector<PalmBox> boxes;

  for (size_t i=0;i<num_boxes;i++){
      const float* ptr = &output_data[i*18];
      if (ptr[0]<score_threshold) continue;
      PalmBox box;
      box.score = ptr[0];
      box.x = ptr[1];
      box.y = ptr[2];
      box.w = ptr[3];
      box.h = ptr[4];
      box.keypoints.assign(ptr+5, ptr+18);
      boxes.push_back(box);
  }


      // 使用 NMS 去掉重复框
  std::vector<PalmBox> final_boxes = nms(boxes, iou_threshold);

  return final_boxes;
}



std::pair<std::vector<float>, std::vector<float>>Detector::infer_output2(const std::vector<float>& input_data,const std::vector<int64_t>& input_shape)
{
    if (!session_) {
        throw std::runtime_error("Session not initialized");
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // 获取输入、输出名字
    auto input_names = session_->GetInputNames();
    auto output_names = session_->GetOutputNames();

    const char* input_name = input_names[0].c_str();
    const char* output_name0 = output_names[0].c_str(); // regression
    const char* output_name1 = output_names[1].c_str(); // scores

    // 创建输入 tensor
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // 推理（两个输出）
    const char* output_names_arr[] = {output_name0, output_name1};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        output_names_arr, 2);

    // 解析 Output 0（regression）
    float* reg_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto reg_shape = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t reg_size = reg_shape.GetElementCount();
    std::vector<float> regress(reg_ptr, reg_ptr + reg_size);

    // 解析 Output 1（scores）
    float* score_ptr = output_tensors[1].GetTensorMutableData<float>();
    auto score_shape = output_tensors[1].GetTensorTypeAndShapeInfo();
    size_t score_size = score_shape.GetElementCount();
    std::vector<float> scores(score_ptr, score_ptr + score_size);

    return {regress, scores};
}

std::vector<PalmBox> Detector::parseOutput_output2(
        const std::vector<float>& regress,
        const std::vector<float>& scores,
        float score_threshold,
        float iou_threshold)
{
    size_t num_boxes = scores.size(); // 2016
    std::vector<PalmBox> boxes;
    boxes.reserve(num_boxes);

    for (size_t i = 0; i < num_boxes; i++) {
        float score = scores[i];
        if (score < score_threshold) continue;

        const float* ptr = &regress[i * 18];  // 每个 box 18 个 float

        PalmBox box;
        box.score = score;

        box.x = ptr[0];
        box.y = ptr[1];
        box.w = ptr[2];
        box.h = ptr[3];

        // 关键点 (14 values)
        box.keypoints.assign(ptr + 4, ptr + 18);

        boxes.push_back(box);
    }

    // 调用 NMS
    return nms(boxes, iou_threshold);
}


// boxes 解码
void Detector::decodeBoxes(const std::vector<float>& raw_boxes,
                 const std::vector<float>& anchors, // 每个 anchor 4 float: cx,cy,w,h
                 std::vector<PalmBox>& boxes_out,
                 int num_boxes,
                 int num_keypoints,
                 float resolution)
{
    boxes_out.clear();
    boxes_out.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        const float* r = &raw_boxes[i * 18]; // 每个 box 18 个
        const float* a = &anchors[i * 4];    // 每个 anchor 4 个

        PalmBox box;

        float x_center = r[0] / resolution * a[2] + a[0];
        float y_center = r[1] / resolution * a[3] + a[1];

        float w = r[2] / resolution * a[2];
        float h = r[3] / resolution * a[3];

        box.x = x_center - w / 2.f; // xmin
        box.y = y_center - h / 2.f; // ymin
        box.w = w;
        box.h = h;

        box.keypoints.resize(num_keypoints * 2);
        for (int k = 0; k < num_keypoints; k++) {
            float kx = r[4 + k * 2] / resolution * a[2] + a[0];
            float ky = r[4 + k * 2 + 1] / resolution * a[3] + a[1];
            box.keypoints[k * 2] = kx;
            box.keypoints[k * 2 + 1] = ky;
        }

        boxes_out.push_back(box);
    }
}

// 根据 raw_score 和 min_score_thresh 筛选并解码
std::vector<PalmBox> Detector::rawOutputToDetections(const std::vector<float>& raw_boxes,
                                           const std::vector<float>& raw_scores,
                                           const std::vector<float>& anchors,
                                           int num_boxes,
                                           int num_keypoints,
                                           float resolution,
                                           float score_threshold)
{
    std::vector<PalmBox> boxes;
    Detector::decodeBoxes(raw_boxes, anchors, boxes, num_boxes, num_keypoints, resolution);

    std::vector<PalmBox> filtered_boxes;
    filtered_boxes.reserve(num_boxes);

    for (int i = 0; i < num_boxes; i++) {
        float score = sigmoid(raw_scores[i]);
        if (score < score_threshold) continue;

        boxes[i].score = score;
        filtered_boxes.push_back(boxes[i]);
    }

    // 调用你的 NMS
    return nms(filtered_boxes, 0.3f);
}


Detector::~Detector(){

  // unique_ptr 会自动调用 delete，释放资源
    if (session_) {
        std::cout << "Releasing ONNX Runtime session...\n";
        session_.reset(); // 显式释放，也可以省略
    }

    if (env_) {
        std::cout << "Releasing ONNX Runtime environment...\n";
        env_.reset(); // 显式释放
    }

}


}