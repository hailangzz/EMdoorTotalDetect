#include <iostream>
#include <sys/time.h>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "event_control.hpp"

using namespace HandDetectRknn;

static ConfigInfo& getConfig()
{
    static ConfigInfo config = readConfig("./config/cfg.txt");  
    return config;
}

static Detector& getDetector()
{
    static Detector detector(getConfig());
    return detector;
}

static HandDetectStateController& getState()
{
    static HandDetectStateController handDetectState(getConfig().max_frame_threshold);
    return handDetectState;
}


bool hand_detect_interface(AndroidImageNV21* image_object_input) {

    Detector& detector = getDetector();
    HandDetectStateController& handDetectState = getState();
  
    std::vector<PalmBox>  results = detector.infer_nv21_zero_copy(image_object_input->image_input_nv21,image_object_input->image_width,image_object_input->image_height);
    bool is_existing_hand = handDetectState.update(results.empty() ? false : true);  //更新手势识别，检测状态
    
    return is_existing_hand;
}
