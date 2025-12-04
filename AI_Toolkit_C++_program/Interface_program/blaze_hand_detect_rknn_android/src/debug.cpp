#include "debug.hpp"
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <iomanip>


#include <android/log.h>
#define LOG_TAG "HandDetectDebug"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)



namespace HandDetectRknn {

// --------------------- 工具：递归创建目录 ---------------------
static bool makeDirsRecursive(const std::string& path)
{
    if (path.empty()) return false;
    if (access(path.c_str(), F_OK) == 0) return true;  // 已经存在

    std::string sub;
    for (size_t i = 1; i < path.size(); ++i)
    {
        if (path[i] == '/')
        {
            sub = path.substr(0, i);
            if (!sub.empty() && access(sub.c_str(), F_OK) != 0)
            {
                if (mkdir(sub.c_str(), 0755) != 0)
                {
                     LOGE("[Debug] Cannot create dir:");
                    return false;
                }
            }
        }
    }

    // 创建最后一级
    if (access(path.c_str(), F_OK) != 0)
    {
        if (mkdir(path.c_str(), 0755) != 0)
        {
            return false;
        }
    }
    LOGI("[Debug] Directory exists or created: %s", path.c_str());
    return true;
}

// --------------------- 构造函数 ---------------------
DebugNv21Saver::DebugNv21Saver(const std::string& save_dir)
        : save_dir_(save_dir), frame_count_(0)
{
    if (!ensureDirectory(save_dir_)) {
        std::cerr << "[Debug] Warning: cannot create directory: " << save_dir_ << std::endl;
        LOGI("[Debug] Warning: cannot create directory: " );
    }
}

// --------------------- 目录检查 ---------------------
bool DebugNv21Saver::ensureDirectory(const std::string& dir)
{
    return makeDirsRecursive(dir);
}

// --------------------- 文件名生成 ---------------------
std::string DebugNv21Saver::generateFileName()
{
    std::ostringstream oss;
    oss << save_dir_
        << "/rgb_"
        << std::setw(5) << std::setfill('0') << frame_count_
        << ".jpg";

    frame_count_++;
    return oss.str();
}

// --------------------- NV21 → JPG 保存函数 ---------------------
bool DebugNv21Saver::saveRgbFrame(const AndroidImageNV21& img)
{
    LOGI("[Debug] NV21 image ptr: %p, width: %d, height: %d", img.image_input_nv21, img.image_width, img.image_height);

    if (img.image_input_nv21 && img.image_width > 0 && img.image_height > 0) {
    const uint8_t* yPlane = img.image_input_nv21;
    const uint8_t* vuPlane = img.image_input_nv21 + img.image_width * img.image_height;

    LOGI("[Debug] First 5 Y values: %d %d %d %d %d",
         yPlane[0], yPlane[1], yPlane[2], yPlane[3], yPlane[4]);
    LOGI("[Debug] First 5 VU values: %d %d %d %d %d",
         vuPlane[0], vuPlane[1], vuPlane[2], vuPlane[3], vuPlane[4]);
    }

    int w = img.image_width;
    int h = img.image_height;

    if (!img.image_input_nv21 || w <= 0 || h <= 0) {
        std::cerr << "[Debug] Invalid NV21 image, skip saving." << std::endl;
        LOGI("[Debug] Invalid NV21 image, skip saving.");
        return false;
    }

    // 1. 构造 YUV Mat，NV21 = Y + VU
    cv::Mat yuv(h + h / 2, w, CV_8UC1, img.image_input_nv21);

    // 2. 转成 RGB/BGR
    cv::Mat rgb;
    cv::cvtColor(yuv, rgb, cv::COLOR_YUV2BGR_NV21);  
    // BGR 是 OpenCV 默认格式，可直接保存 jpg

    LOGI("[Debug] RGB Mat size: %dx%d, channels: %d, type: %d",
     rgb.cols, rgb.rows, rgb.channels(), rgb.type());

    // 打印前几个像素值
    LOGI("[Debug] RGB first pixel: %d %d %d", rgb.data[0], rgb.data[1], rgb.data[2]);

    // 3. 生成 jpg 文件名
    std::string filename = generateFileName();
    LOGI("[Debug] Saving RGB frame to: %s", filename.c_str());

    // 4. 保存 JPEG 文件
    if (!cv::imwrite(filename, rgb)) {
        std::cerr << "[Debug] Failed to write JPG file: " << filename << std::endl;
        LOGI("[Debug] Failed to write JPG file:");
        return false;
    }

    std::cout << "[Debug] Saved frame => " << filename << std::endl;
    LOGI("[Debug] Saved RGB frame as JPG → %s", filename.c_str());
    return true;
}

} // namespace HandDetectRknn
