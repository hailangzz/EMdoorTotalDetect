#ifndef DEBUG_NV21_SAVER_HPP
#define DEBUG_NV21_SAVER_HPP

#include <string>
#include "types.hpp"

namespace HandDetectRknn {

class DebugNv21Saver {
public:
    explicit DebugNv21Saver(const std::string& save_dir);

    // 将 NV21 保存为 JPG
    bool saveRgbFrame(const AndroidImageNV21& img);

private:
    bool ensureDirectory(const std::string& dir);
    std::string generateFileName();

private:
    std::string save_dir_;
    int frame_count_;
};

}

#endif
