import time
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge

from rgb_subscriber import RGBDataSubscriber



def yuv420_to_rgb(data_bytes, width, height):
    """
    data_bytes: bytes 或 bytearray，YUV420 (I420) 格式
    width, height: 图像尺寸
    返回: BGR numpy 数组 (H, W, 3)
    """
    yuv = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height * 3 // 2, width))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
    return rgb



def main():
    rospy.init_node("shelf_rgb_ai", anonymous=True)
    img_pub = rospy.Publisher("/shelf_processed_img", Image, queue_size=1)
    detect_pub = rospy.Publisher("/detect_hand", Empty, queue_size=1)
    bridge = CvBridge()

    subscriber = RGBDataSubscriber(topic_name="RgbTopic", domain_id=0)

    try:
        while not rospy.is_shutdown():
            latest = subscriber.get_latest()
            if latest is not None:
                if len(latest["data_ptr"]) == 0:
                    print("image data is null，waitting for next frame...")
                    continue
                width = latest["width"]
                height = latest["height"]
                data_bytes = bytes(latest["data_ptr"])
                #获取到rgb_frame,用来AI处理
                rgb_frame = yuv420_to_rgb(data_bytes, width, height)
                # print(rgb_frame)
                """
                此处编写AI检测代码
                """
                #如果检测到手,则发布
                #detect_pub.publish(Empty())
                    

                #发布处理后图像
                #processed_img = bridge.cv2_to_imgmsg(processed_rgb_frame, encoding="rgb8")
                #img_pub.publish(processed_img)

            time.sleep(0.03)  # 30ms，约33FPS
    except KeyboardInterrupt:
        print("用户中断，退出...")
    finally:
        subscriber.close()

if __name__ == "__main__":
    main()

