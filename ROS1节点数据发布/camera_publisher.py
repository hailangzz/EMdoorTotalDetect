#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os,sys

# 确保 Python 能找到 ROS 的库
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

# 确保系统能找到 .so 动态库
os.environ['LD_LIBRARY_PATH'] = '/opt/ros/noetic/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

def main():
    # 初始化 ROS 节点
    rospy.init_node('camera_publisher', anonymous=True)

    # 创建 Publisher
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    # 创建 CvBridge，用于 OpenCV 图像和 ROS Image 消息转换
    bridge = CvBridge()


    # # 打开摄像头 (0 = 默认摄像头)
    videos_path = r"/home/chenkejing/Videos/test_videos/HandsDance.mp4"
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videos_path)
    if not cap.isOpened():
        rospy.logerr("Cannot open camera")
        return

    rospy.loginfo("Camera publisher started, publishing to /camera/image_raw")
    rate = rospy.Rate(30)  # 发布帧率 30 Hz

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to capture frame")
            continue

        # 将 OpenCV 图像转换为 ROS Image 消息
        msg = bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        # 发布
        pub.publish(msg)

        # 控制帧率
        rate.sleep()

    # 程序退出前释放摄像头
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
