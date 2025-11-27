#include "image_listener.hpp"

ImageListener::ImageListener(ros::NodeHandle& nh)
    : it_(nh)
{
    sub_ = it_.subscribe("camera/image", 1, &ImageListener::imageCallback, this);
    ROS_INFO("ImageListener subscribed to camera/image");
}

void ImageListener::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        // 解码为 YUV420
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "yuv420");

        // 转为 BGR
        cv::Mat bgr;
        cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_YUV2BGR_I420);

        // 调整尺寸到 192×192
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(192, 192));

        // 存储
        {
            std::lock_guard<std::mutex> lock(mtx_);
            latest_bgr_ = resized.clone();
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

bool ImageListener::getLatestBGR(cv::Mat& out)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (latest_bgr_.empty())
        return false;

    out = latest_bgr_.clone();
    return true;
}
