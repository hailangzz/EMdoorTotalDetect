import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge

from rgb_subscriber import RGBDataSubscriber
import cv2
import time
import blazepalm_utils as but
import RKNNInference
import debug

import os
import logging
import logging.config
# 屏蔽 ROS 自动加载日志配置
os.environ['ROSCONSOLE_CONFIG_FILE'] = '/dev/null'

# 手动重置 Python logging 系统，防止 rospy 再次注册 fileConfig
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,  # 可改成 logging.DEBUG / logging.WARNING
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# 强制确保 logging 模块中的 level=DEBUG 是数字级别（兼容 ROS）
logging._levelToName.update({10: 'DEBUG', 20: 'INFO', 30: 'WARNING', 40: 'ERROR', 50: 'CRITICAL'})
logging._nameToLevel.update({'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40, 'CRITICAL': 50})




def main():

    rospy.init_node("shelf_rgb_ai", anonymous=True)
    img_pub = rospy.Publisher("/shelf_processed_img", Image, queue_size=1)
    detect_pub = rospy.Publisher("/detect_hand", Empty, queue_size=1)
    bridge = CvBridge()
    subscriber = RGBDataSubscriber(topic_name="RgbTopic", domain_id=0)

    BlazePalmModelInfo = {"IMAGE_HEIGHT": 192,
                          "IMAGE_WIDTH": 192,
                          "CHANNEL_FIRST": False,
                          "RKNN_MODEL": "palm_detection_full.rknn",
                          "ANCHOR_PATH": "anchors_192.npy"
                          }
    controller = but.HandStateController(threshold=5)
    rknn_infer = RKNNInference.RKNNInference(BlazePalmModelInfo)
    detect_hand = False

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
                rgb_frame = but.yuv420_to_rgb(data_bytes, width, height)


                # print(rgb_frame)
                """
                此处编写AI检测代码                
                """
                outputs, frame = rknn_infer.infer(rgb_frame)
                normalized_detections = but.postprocess(outputs, anchor_path=BlazePalmModelInfo["ANCHOR_PATH"],
                                                        resolution=BlazePalmModelInfo["IMAGE_WIDTH"])[0]
                detections = but.denormalize_detections(normalized_detections,
                                                        rknn_infer.image_resize_pad_info["scale"],
                                                        rknn_infer.image_resize_pad_info["pad"],
                                                        resolution=BlazePalmModelInfo["IMAGE_WIDTH"])


                if detections.shape[0]>=1:
                    print("detect hand object numbers %d!",detections.shape[0])

                    detect_hand = True
                else:
                    detect_hand = False

                state = controller.update(detect_hand)
                if state:
                    result_img = but.display_result(frame, detections)  # 添加检测框，到图像数据中
                    debug.save_frame_rgb_image(result_img)
                    # 如果检测到手,则发布
                    detect_pub.publish(Empty())
                    # 发布处理后图像
                    processed_img = bridge.cv2_to_imgmsg(result_img, encoding="rgb8")
                    img_pub.publish(processed_img)
                    print('detect hand object!!')
                else:
                    print('null of hand detect')

            time.sleep(0.03)  # 30ms，约33FPS
    except KeyboardInterrupt:
        print("用户中断，退出...")
    finally:
        subscriber.close()

if __name__ == "__main__":
    main()

