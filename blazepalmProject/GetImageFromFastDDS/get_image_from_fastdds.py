import os
import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # 改成你的Topic名
            self.listener_callback,
            10)
        self.subscription

        # 保存图片的文件夹
        self.save_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(self.save_dir, exist_ok=True)

        # 控制保存逻辑
        self.last_save_time = 0
        self.save_interval = 3.0  # 每3秒保存一次
        self.image_count = 0
        self.max_images = 10

    def listener_callback(self, msg):
        # 将ROS图像消息转为OpenCV格式
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 检查是否到达保存时间
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_interval:
            self.image_count += 1
            filename = os.path.join(self.save_dir, f"frame_{self.image_count:02d}.jpg")
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"✅ Saved image {self.image_count}: {filename}")
            self.last_save_time = current_time

            # 达到最大张数后退出
            if self.image_count >= self.max_images:
                self.get_logger().info("📸 Captured 10 images, shutting down.")
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
