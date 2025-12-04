
#include <sys/time.h>
#include "blaze_hand_detect_rknn.hpp"  // 上面写好的 Detector 类
#include "event_control.hpp"
#include "debug.hpp"
#include <atomic>

#define ANDROID_ENV 1

#if ANDROID_ENV
    #include <android/log.h>
    #define LOG_TAG "HandDetectNative"
    #define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
    #define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
    #define LOGI(...) std::cout << "[INFO] " << __VA_ARGS__ << std::endl
    #define LOGE(...) std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl
#endif


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

static DebugNv21Saver& getDebugSaver() {
    static DebugNv21Saver saver(getConfig().debug_nv21_image_saver);
    return saver;
}

bool hand_detect_interface(AndroidImageNV21* image_object_input) {
    static std::atomic<bool> is_running(false); // 运行状态标志

    // 如果上一次还在执行，直接丢弃输入
    if (is_running.load()) {
        return false;
    }

    // 标记开始运行
    is_running.store(true);

    LOGI("Enter hand_detect_interface");
    std::cout << "enter hand_detect_interface function ：" << std::endl;

    if (!image_object_input || !image_object_input->image_input_nv21) {
        std::cerr << "Invalid NV21 image input!" << std::endl;
        LOGE("Invalid NV21 image input!");
        is_running.store(false);  // 记得清掉标志
        return false;
    }

    if (image_object_input->image_width <= 0 || image_object_input->image_height <= 0) {
        std::cerr << "Invalid image size!" << std::endl;
        LOGE("Invalid image size!");
        is_running.store(false);
        return false;
    }

    getDebugSaver().saveRgbFrame(*image_object_input);

    Detector& detector = getDetector();
    HandDetectStateController& handDetectState = getState();
    
    std::cout << "Start Hand Detect Model Inference：" << std::endl;
    LOGI("Start Hand Detect Model Inference");

    std::vector<PalmBox> results = detector.infer_nv21_zero_copy_BGR(
        image_object_input->image_input_nv21,
        image_object_input->image_width,
        image_object_input->image_height
    );

    std::cout << "Hand Detect Model Inference is finished." << std::endl;
    LOGI("Hand Detect Model Inference finished, result count = %zu", results.size());

    bool is_existing_hand = handDetectState.update(!results.empty());  // 更新手势检测状态
    LOGI("Hand state updated, existing_hand = %d", is_existing_hand);

    // 标记执行结束
    is_running.store(false);

    return is_existing_hand;
}