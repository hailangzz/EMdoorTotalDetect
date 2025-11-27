#pragma once
#include "video_reader.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "blaze_hand_detect.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <ros/ros.h>
#include <thread>

#include "image_listener.hpp"

using namespace HandDetect;


int main(int argc, char** argv)
{
    const std::string config_file = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/config/cfg.txt";
    ConfigInfo cfg_values;  
    auto cfg = readConfig(config_file,cfg_values); 
     
    
    std::vector<float> anchors = loadAnchorsBin(cfg_values.anchors_path); // 读取anchors信息

    VideoInfo videos_infos;
    videos_infos.video_path = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/videos/HandsDance1.mp4";
    use_utils_read_videos_info(videos_infos);

    Detector hand_detect;
    // const char* model_path = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/models/palm_detection_full.onnx";
    hand_detect.loadModel(cfg_values.model_path);
    hand_detect.printModelInfo();

    std::vector<float> output;    
    output = hand_detect.test_matrix_info(hand_detect);
    
    std::cout << "Output size: " << output.size() << std::endl;
    std::cout << "First 10 outputs: ";
    for (size_t i = 0; i < 10 && i < output.size(); i++) std::cout << output[i] << " ";
    std::cout << std::endl;

    std::string image_path = "videos/hand.png";
    std::vector<float> input_data;
    std::vector<int64_t> input_shape;
    if (!resizeImage(image_path, input_data, input_shape)) {
        return -1;
    }

    // 2. 推理
    auto [raw_boxes, raw_scores] = hand_detect.infer_output2(input_data, input_shape);

    // 4. 调用后处理函数，解码输出并进行 NMS
    std::vector<PalmBox> boxes = hand_detect.rawOutputToDetections(
        raw_boxes, raw_scores, anchors, cfg_values.num_boxes, cfg_values.num_keypoints, cfg_values.resolution, cfg_values.score_threshold);


    // 接入ros topic数据

    ros::init(argc, argv, "hand_detector_node");
    ros::NodeHandle nh;

    ImageListener listener(nh);
    // 让 ROS 回调在独立线程中运行
    std::thread ros_thread([]() {
        ros::spin();
    });

    ros::Rate rate(30); // 推理频率
    cv::Mat frame;

    while (ros::ok())
    {
        if (listener.getLatestBGR(frame))
        {
            // frame 是 BGR 192×192，直接送入 ONNX 模型
            auto [raw_boxes, raw_scores] = hand_detect.infer_output2(frame, input_shape);
            std::vector<PalmBox> boxes = hand_detect.rawOutputToDetections(
        raw_boxes, raw_scores, anchors, cfg_values.num_boxes, cfg_values.num_keypoints, cfg_values.resolution, cfg_values.score_threshold);


        }

        rate.sleep();
    }

    ros_thread.join();
    // plotDetectBoxs(image_path,boxes);


    return 0;
}
