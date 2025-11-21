#pragma once
#include "video_reader.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "blaze_hand_detect.hpp"
#include "types.hpp"
#include "utils.hpp"


using namespace HandDetect;


int use_video_reader() {

    std::string video_path = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/videos/HandsDance1.mp4";
    VideoReader reader(video_path);

    std::cout << "Video opened: " << video_path << std::endl;
    std::cout << "Width: " << reader.getWidth() << ", Height: " << reader.getHeight() << ", FPS: " << reader.getFPS() << std::endl;

    cv::Mat frame;
    while (reader.getNextFrame(frame)) {
        if (frame.empty()) break;

        // 显示帧
        cv::imshow("Video", frame);
        if (cv::waitKey(1000 / static_cast<int>(reader.getFPS())) == 27) break; // 按 ESC 退出
    }

    std::cout << "Video playback finished." << std::endl;
    return 0;

}

void use_utils_read_videos_info(VideoInfo& videos_infos)
{
    if (!GetVideoInfo(videos_infos.video_path,
                      videos_infos.width,
                      videos_infos.height,
                      videos_infos.fps,
                      videos_infos.frame_count)) {

        std::cerr << "Failed to read video info." << std::endl;
        return;
    }

    std::cout << "VideoInfo:" << std::endl;
    std::cout << "  video_path:  " << videos_infos.video_path << std::endl;
    std::cout << "  width:       " << videos_infos.width << std::endl;
    std::cout << "  height:      " << videos_infos.height << std::endl;
    std::cout << "  fps:         " << videos_infos.fps << std::endl;
    std::cout << "  frame_count: " << videos_infos.frame_count << std::endl;
}


std::vector<float> test_matrix_info(Detector & hand_detect){
    // 假设输入张量为 [1,3,224,224]
    std::vector<float> input_data(1*192*192*3, 0.5f); // 示例数据
    std::vector<int64_t> input_shape = {1, 192, 192, 3};  
    std::vector<float> output = hand_detect.infer(input_data, input_shape);

    return output;

    
}

int main(int argc, char** argv)
{
    
    VideoInfo videos_infos;
    videos_infos.video_path = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/videos/HandsDance1.mp4";
    use_utils_read_videos_info(videos_infos);

    Detector hand_detect;
    const char* model_path = "/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/models/palm_detection_full.onnx";
    hand_detect.loadModel(model_path);
    hand_detect.printModelInfo();

    std::vector<float> output;    
    output = test_matrix_info(hand_detect);

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

    // output = hand_detect.infer(input_data, input_shape);
    // std::cout << "Output size: " << output.size() << std::endl;
    // std::cout << "First 10 outputs: ";
    // for (size_t i = 0; i < 10 && i < output.size(); i++) std::cout << output[i] << " ";
    // std::cout << std::endl;
    
    // std::vector<PalmBox> boxes = hand_detect.parseOutput(output, 0.7f, 0.3f);

    // for (const auto& box : boxes) {
    //     std::cout << "score=" << box.score
    //             << ", bbox=(" << box.x << "," << box.y
    //             << "," << box.w << "," << box.h << ")\n";
    // }

    auto [regress, scores]  = hand_detect.infer_output2(input_data, input_shape);
    std::cout << "Output size: " << output.size() << std::endl;
    std::cout << "First 10 outputs: ";
    for (size_t i = 0; i < 10 && i < output.size(); i++) std::cout << output[i] << " ";
    std::cout << std::endl;
    
    auto boxes = hand_detect.parseOutput_output2(regress,scores, 0.7f, 0.3f);

    for (const auto& box : boxes) {
        std::cout << "score=" << box.score
                << ", bbox=(" << box.x << "," << box.y
                << "," << box.w << "," << box.h << ")\n";
    }




    // 2. 推理
    auto [raw_boxes, raw_scores] = hand_detect.infer_output2(input_data, input_shape);

    // 3. 准备 anchors
    // 假设你已经生成 anchors 数组，长度 = num_boxes * 4
    // std::vector<float> anchors;
    std::vector<float> anchors = loadAnchorsBin("/home/chenkejing/git_director/lubancat_ai_manual_code/dev_env/onnx_runtime/blazepalm_onnxruntime_cpp/models/anchors_192.bin");
    int num_boxes = 2016;      // 你的模型输出框数量
    int num_keypoints = 7;     // BlazePalm 每个手 7 个关键点
    float resolution = 192.f;  // 输入图片尺寸
    float score_threshold = 0.7f;

    // anchors 可以通过预先生成或从模型配置读取
    // anchors.resize(num_boxes * 4);

    // 4. 调用后处理函数，解码输出并进行 NMS
    boxes = hand_detect.rawOutputToDetections(
        raw_boxes, raw_scores, anchors, num_boxes, num_keypoints, resolution, score_threshold);

    // 5. 打印结果
    for (size_t i = 0; i < boxes.size(); i++) {
        const PalmBox& box = boxes[i];
        std::cout << "Box " << i 
                  << ": score=" << box.score 
                  << " x=" << box.x 
                  << " y=" << box.y 
                  << " w=" << box.w 
                  << " h=" << box.h 
                  << std::endl;
        std::cout << "  keypoints: ";
        for (float kp : box.keypoints) std::cout << kp << " ";
        std::cout << std::endl;
    }

    cv::Mat image = cv::imread("videos/hand.png");
    // 绘制检测结果
    drawPalmBoxes(image, boxes);

    // 显示图像
    cv::imshow("Palm Detection", image);
    cv::waitKey(0);

    // 可选：保存到文件
    cv::imwrite("palm_detect_result.jpg", image);

    
    return 0;
}
