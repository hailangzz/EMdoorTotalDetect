#include "video_reader.hpp"


VideoReader::VideoReader(const std::string& video_path) {
    cap_.open(video_path);
    if (!cap_.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        throw std::runtime_error("Cannot open video file");
    }
}

VideoReader::~VideoReader() {
    cap_.release();
}

bool VideoReader::getNextFrame(cv::Mat& frame) {
    return cap_.read(frame);  // 返回 false 表示读完
}

int VideoReader::getWidth() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
}

int VideoReader::getHeight() const {
    return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
}

double VideoReader::getFPS() const {
    return cap_.get(cv::CAP_PROP_FPS);
}
