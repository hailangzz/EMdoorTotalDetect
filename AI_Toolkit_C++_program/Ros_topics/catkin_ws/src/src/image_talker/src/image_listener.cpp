#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

// 回调函数，每收到一帧图像就调用
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        // 将 ROS 图像消息转为 OpenCV 图像
        cv::Mat frame = cv_bridge::toCvShare(msg, "bgr8")->image;

        // 显示图像
        cv::imshow("Received Image", frame);
        cv::waitKey(1); // 必须调用才能刷新窗口
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_listener"); // 节点名称
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("camera/image", 1, imageCallback);

    ROS_INFO("Subscribed to /camera/image topic, waiting for images...");

    ros::spin(); // 循环等待回调
    return 0;
}
