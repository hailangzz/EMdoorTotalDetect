#ifndef IMAGE_LISTENER_H
#define IMAGE_LISTENER_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <opencv2/opencv.hpp>

class ImageListener
{
public:
    ImageListener(ros::NodeHandle& nh);

    // 获取最新的 BGR 图像（已经 resize 到 192×192）
    bool getLatestBGR(cv::Mat& out);

private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;

    std::mutex mtx_;
    cv::Mat latest_bgr_;   // 保存最新的 BGR 图像（192×192）

    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
};

#endif
