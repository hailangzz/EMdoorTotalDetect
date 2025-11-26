#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "image_talker");
    ros::NodeHandle nh;

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub = it.advertise("camera/image", 1);

    // 读取本地图片
    std::string image_path = "/home/chenkejing/images/test.jpg"; // 替换成你本地图片路径
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        ROS_ERROR("Cannot read image at %s", image_path.c_str());
        return -1;
    }

    ros::Rate loop_rate(10); // 10 FPS
    while (ros::ok()) {
        // 转为 ROS Image 消息
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();

        pub.publish(msg);
        ROS_INFO("Published image");

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
