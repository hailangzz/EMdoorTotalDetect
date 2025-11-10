import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from rgb_subscriber import RGBDataSubscriber
import cv2
import time
import blazepalm_utils as but
import RKNNInference
import debug

import os
import configparser
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


class HandDetect:
    def __init__(self,):
        self.subscriber = None
        self.logger = debug.ProjectDebug()
        self.BlazePalmModelInfo = {"IMAGE_HEIGHT": 192,
                          "IMAGE_WIDTH": 192,
                          "CHANNEL_FIRST": False,
                          "RKNN_MODEL": "palm_detection_full.rknn",
                          "ANCHOR_PATH": "anchors_192.npy"
                          }
        self.config_path = r"./config.ini"
        self.controller = but.HandStateController(threshold=5)
        self.rknn_infer = RKNNInference.RKNNInference(self.BlazePalmModelInfo)
        self.bgr_origin_image_crop_config = {
                                        "center_x": 130,
                                        "center_y": 117,
                                        "axes_w": 92,
                                        "axes_h": 95}
        self.read_config_info()

        self.img_pub = None
        self.detect_pub = None
        self.bridge = None
        self.subscriber = None
        self.create_rospy_topics()

        self.send_topic_time_thread = 1.0
        self.last_send_topic_time = time.time()
        self.last_hand_detect_state = False

    def read_config_info(self):

        if os.path.exists(self.config_path):
            config = configparser.ConfigParser()
            config.read(self.config_path, encoding='utf-8')
            self.bgr_origin_image_crop_config = {
                "center_x": config.getint("ImageCropConfig", "center_x"),
                "center_y": config.getint("ImageCropConfig", "center_y"),
                "axes_w": config.getint("ImageCropConfig", "axes_w"),
                "axes_h": config.getint("ImageCropConfig", "axes_h")
            }

    def create_rospy_topics(self):
        rospy.init_node("shelf_rgb_ai", anonymous=True)
        self.img_pub = rospy.Publisher("/shelf_processed_img", Image, queue_size=1)
        self.detect_pub = rospy.Publisher("/detect_hand", Bool, queue_size=1)
        self.bridge = CvBridge()
        self.subscriber = RGBDataSubscriber(topic_name="RgbTopic", domain_id=0)

    def infer_detect(self):
        try:

            while not rospy.is_shutdown():
                latest = self.subscriber.get_latest()
                if latest is not None:
                    if len(latest["data_ptr"]) == 0:
                        print("image data is null，waitting for next frame...")
                        continue
                    width = latest["width"]
                    height = latest["height"]
                    data_bytes = bytes(latest["data_ptr"])
                    # 获取到rgb_frame,用来AI处理
                    origin_bgr_image = but.yuv420_to_rgb(data_bytes, width, height)
                    self.rknn_infer.infer_image_data = but.image_crop_ellipse(origin_bgr_image,
                                                                              self.bgr_origin_image_crop_config)

                    self.rknn_infer.infer()
                    self.post_processing()
                    self.controller_hand_detect_state()
                else:
                    pass
                    # self.logger.save_detect_object_log("subscriber.get_latest() is None!\n")

        except KeyboardInterrupt:
            print("user interrupt，quit...")
        finally:
            self.subscriber.close()

    def post_processing(self):
        normalized_detections = but.postprocess(self.rknn_infer.outputs,
                                                anchor_path=self.BlazePalmModelInfo["ANCHOR_PATH"],
                                                resolution=self.BlazePalmModelInfo["IMAGE_WIDTH"])[0]
        self.rknn_infer.detections = but.denormalize_detections(normalized_detections,
                                                self.rknn_infer.image_resize_pad_info["scale"],
                                                self.rknn_infer.image_resize_pad_info["pad"],
                                                resolution=self.BlazePalmModelInfo["IMAGE_WIDTH"])

        self.rknn_infer.result_img = but.display_result(self.rknn_infer.infer_image_data,
                                                        self.rknn_infer.detections)  # 添加检测框，到图像数据中
        # print("detect object number:"+str(self.rknn_infer.detections.shape[0]))

    def controller_hand_detect_state(self):
        # print("controller_hand_detect_state detect object number:" + str(self.rknn_infer.detections.shape[0]))
        if self.rknn_infer.detections.shape[0] >= 1:
            detect_hand_bool = True
            # self.logger.save_detect_object_log("detect hand object number : "+str(self.rknn_infer.detections.shape[0])+"\n")
            # self.logger.save_frame_rgb_image(self.rknn_infer.result_img)
        else:
            detect_hand_bool = False

        self.controller.update(detect_hand_bool)   # 更新目标检测结果状态
        self.send_hand_detcet_event_info()        # 发送目标检测结果状态信息

    def send_hand_detcet_event_info(self):
        # 作用：每间隔1秒钟，发送一次手部检测结果的状态值

        current_time = time.time()
        delta_t = current_time - self.last_send_topic_time
        # print('the delta_t is :%f' % delta_t)
        if delta_t > self.send_topic_time_thread:
            # self.logger.save_detect_object_log("detect hand object number : "+str(self.rknn_infer.detections.shape[0])+"\n")
            # self.logger.save_frame_rgb_image(self.rknn_infer.result_img)

            if self.controller.event_state:
                # 如果检测到手,则发布
                self.detect_pub.publish(Bool(self.controller.event_state))
                self.last_hand_detect_state = self.controller.event_state
                self.logger.save_detect_object_log(
                    "detect hand object number : " + str(self.rknn_infer.detections.shape[0]) + "\n")
            else:
                if self.last_hand_detect_state:
                    self.detect_pub.publish(Bool(self.controller.event_state))
                    self.last_hand_detect_state = self.controller.event_state
                    self.logger.save_detect_object_log(
                        "detect hand object number : " + str(self.rknn_infer.detections.shape[0]) + "\n")

            # # 发布处理后图像
            # processed_img = self.bridge.cv2_to_imgmsg(self.rknn_infer.result_img, encoding="rgb8")
            # self.img_pub.publish(processed_img)
            self.last_send_topic_time = current_time

    def __del__(self):
        self.subscriber.close()


def detect_program():
    logger = debug.ProjectDebug()

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

    # detct_umbers = 0
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
                # 获取到rgb_frame,用来AI处理
                rknn_infer.infer_image_data = but.yuv420_to_rgb(data_bytes, width, height)
                """
                此处编写AI检测代码                
                """
                rknn_infer.infer()
                normalized_detections = but.postprocess(rknn_infer.outputs, anchor_path=BlazePalmModelInfo["ANCHOR_PATH"],
                                                        resolution=BlazePalmModelInfo["IMAGE_WIDTH"])[0]
                detections = but.denormalize_detections(normalized_detections,
                                                        rknn_infer.image_resize_pad_info["scale"],
                                                        rknn_infer.image_resize_pad_info["pad"],
                                                        resolution=BlazePalmModelInfo["IMAGE_WIDTH"])

                result_img = but.display_result(rknn_infer.infer_image_data, detections)  # 添加检测框，到图像数据中

                if detections.shape[0] >= 1:
                    detect_hand = True
                else:
                    detect_hand = False

                controller.update(detect_hand)
                if controller.state:
                    # 如果检测到手,则发布
                    detect_pub.publish(Empty())
                    # 发布处理后图像
                    processed_img = bridge.cv2_to_imgmsg(result_img, encoding="rgb8")
                    img_pub.publish(processed_img)
                    print('detect hand object event is success!!')
                # else:
                #     print('null of hand detect')

            # time.sleep(0.03)  # 30ms，约33FPS
    except KeyboardInterrupt:
        print("用户中断，退出...")
    finally:
        subscriber.close()


if __name__ == "__main__":
    detect_program()

