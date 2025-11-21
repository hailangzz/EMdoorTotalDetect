#include <string>
#include <opencv2/opencv.hpp>

class VideoReader {
public:
    VideoReader(const std::string& video_path);
    ~VideoReader();

    // 读取下一帧，返回 false 表示视频结束
    bool getNextFrame(cv::Mat& frame);

    // 获取视频帧宽、高、FPS
    int getWidth() const;
    int getHeight() const;
    double getFPS() const;

private:
    cv::VideoCapture cap_;
};
