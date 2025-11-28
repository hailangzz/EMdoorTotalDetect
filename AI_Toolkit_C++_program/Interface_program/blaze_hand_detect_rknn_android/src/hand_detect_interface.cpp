
#include <sys/time.h>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "event_control.hpp"

using namespace HandDetectRknn;


// 保存配置文件路径
static std::string& getConfigPath() {
    static std::string config_path = "./config/cfg.txt";  // 默认路径
    return config_path;
}


// 配置读取函数
static ConfigInfo& getConfig()
{
    static ConfigInfo config = readConfig(getConfigPath().c_str());
    return config;
}


// ⭐给调用者的接口：设置配置文件路径
void hand_detect_set_config_path(const std::string& path) {
    getConfigPath() = path;
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

    if (!image_object_input || !image_object_input->image_input_nv21) {
        std::cerr << "Invalid NV21 image input!" << std::endl;
        return false;
    }

    if (image_object_input->image_width <= 0 || image_object_input->image_height <= 0) {
        std::cerr << "Invalid image size!" << std::endl;
        return false;
    }



    Detector& detector = getDetector();
    HandDetectStateController& handDetectState = getState();
  
    std::vector<PalmBox>  results = detector.infer_nv21_zero_copy(image_object_input->image_input_nv21,image_object_input->image_width,image_object_input->image_height);
    bool is_existing_hand = handDetectState.update(results.empty() ? false : true);  //更新手势识别，检测状态
    
    return is_existing_hand;
}
