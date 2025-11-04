import cv2
import numpy as np
from rknn.api import RKNN
import blazepalm_utils as but

class RKNNInference:
    def __init__(self, model_info):
        """
        初始化 RKNN 推理类
        :param model_path: RKNN 模型路径
        :param input_size: 输入尺寸 (宽, 高)
        :param mean_values: 均值，用于归一化
        :param std_values: 标准差，用于归一化
        """
        self.rknn_config = {"target_platform": 'rk3588',
                            "mean_values": [[0, 0, 0]],
                            "std_values": [[255, 255, 255]],
                            }


        self.model_path = model_info["RKNN_MODEL"]
        self.input_height_size = model_info["IMAGE_HEIGHT"]
        self.input_width_size = model_info["IMAGE_WIDTH"]
        self.image_channel_first = model_info["CHANNEL_FIRST"]
        self.image_resize_pad_info = {"scale":0,"pad":0}


        self.rknn = RKNN()
        # 初始化模型
        self._load_model()

    def _load_model(self):
        print(f"load rknn model: {self.model_path}")
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load rknn model is fail: {ret}")

        # 初始化 runtime
        ret = self.rknn.init_runtime(target=self.rknn_config["target_platform"])
        if ret != 0:
            raise RuntimeError(f"init rknn runtime is fail: {ret}")

        print("RKNN init succeed!")


    def _preprocess(self, frame):
        """预处理摄像头图像"""

        img256, _, scale, pad = but.resize_pad(frame[:, :, ::-1], self.input_height_size)
        input_data = img256.astype('float32')
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)
        if not self.image_channel_first:
            input_data = input_data.transpose((0, 2, 3, 1))

        self.image_resize_pad_info["scale"]= scale
        self.image_resize_pad_info["pad"] = pad

        return input_data

    def infer(self,frame_rgb_image_data):
        """从摄像头读取一帧并推理"""
        if frame_rgb_image_data is None:
            raise RuntimeError("fastdds frame image is error!")

        input_data = self._preprocess(frame_rgb_image_data)
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs, frame_rgb_image_data

    def release(self):
        """释放RKNN 资源"""

        if self.rknn:
            self.rknn.release()
            self.rknn = None
        print("资源释放完成")

    def __del__(self):
        self.release()