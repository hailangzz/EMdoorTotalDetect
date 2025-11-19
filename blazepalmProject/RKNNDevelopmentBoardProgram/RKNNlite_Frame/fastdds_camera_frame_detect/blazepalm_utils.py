import cv2
import numpy as np
from scipy.special import expit
import os
import time
import math

num_coords = 18
min_score_thresh = 0.7
min_suppression_threshold = 0.3
num_keypoints = 7


def resize_pad(img, resolution):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0] >= size0[1]:
        h1 = resolution
        w1 = resolution * size0[1] // size0[0]
        padh = 0
        padw = resolution - w1
        scale = size0[1] / w1
    else:
        h1 = resolution * size0[0] // size0[1]
        w1 = resolution
        padh = resolution - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh // 2
    padh2 = padh // 2 + padh % 2
    padw1 = padw // 2
    padw2 = padw // 2 + padw % 2
    img1 = cv2.resize(img, (w1, h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0, 0)), mode='constant')
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (resolution, resolution))
    return img1, img2, scale, pad


def decode_boxes(raw_boxes, anchors, resolution):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    x_scale = resolution
    y_scale = resolution
    h_scale = resolution
    w_scale = resolution

    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(num_keypoints):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def raw_output_to_detections(raw_box, raw_score, anchors, resolution):
    """The output of the neural network is an array of shape (b, 896, 18)
    containing the bounding box regressor predictions, as well as an array
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" arrays into proper detections.
    Returns a list of (num_detections, 13) arrays, one for each image in
    the batch.

    This is based on the source code from:
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
    """
    detection_boxes = decode_boxes(raw_box, anchors, resolution)

    thresh = 100.0
    raw_score = raw_score.clip(-thresh, thresh)
    # instead of defining our own sigmoid function which yields a warning)
    # expit = sigmoid
    detection_scores = expit(raw_score).squeeze(axis=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= min_score_thresh

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = np.expand_dims(detection_scores[i, mask[i]], axis=-1)
        output_detections.append(np.concatenate((boxes, scores), axis=-1))

    return output_detections


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = np.minimum(
        np.repeat(np.expand_dims(box_a[:, 2:], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, 2:], axis=0), A, axis=0),
    )
    min_xy = np.maximum(
        np.repeat(np.expand_dims(box_a[:, :2], axis=1), B, axis=1),
        np.repeat(np.expand_dims(box_b[:, :2], axis=0), A, axis=0),
    )
    inter = np.clip((max_xy - min_xy), 0, None)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = np.repeat(
        np.expand_dims(
            (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]),
            axis=1
        ),
        inter.shape[1],
        axis=1
    )  # [A,B]
    area_b = np.repeat(
        np.expand_dims(
            (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]),
            axis=0
        ),
        inter.shape[0],
        axis=0
    )  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)


def weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    # argsort() returns ascending order, therefore read the array from end
    remaining = np.argsort(detections[:, num_coords])[::-1]

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords + 1]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def denormalize_detections(detections, scale, pad, resolution):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    image_size = resolution

    detections[:, 0] = detections[:, 0] * scale * image_size - pad[0]
    detections[:, 1] = detections[:, 1] * scale * image_size - pad[1]
    detections[:, 2] = detections[:, 2] * scale * image_size - pad[0]
    detections[:, 3] = detections[:, 3] * scale * image_size - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * image_size - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * image_size - pad[0]
    return detections


def postprocess(preds_ailia, anchor_path='anchors.npy', resolution=256):
    """
    Process detection predictions from ailia and return filtered detections
    """
    raw_box = preds_ailia[0]  # (1, 896, 18)
    raw_score = preds_ailia[1]  # (1, 896, 1)

    anchors = np.load(anchor_path).astype("float32")

    # Postprocess the raw predictions:
    detections = raw_output_to_detections(raw_box, raw_score, anchors, resolution)

    # Non-maximum suppression to remove overlapping detections:
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, num_coords + 1))
        filtered_detections.append(faces)

    return filtered_detections


def get_savepath(arg_path, src_path, prefix='', post_fix='_res', ext=None):
    if '.' in arg_path:
        arg_base, arg_ext = os.path.splitext(arg_path)
        new_ext = arg_ext if ext is None else ext
        new_path = arg_base + new_ext
    else:
        src_base, src_ext = os.path.splitext(os.path.basename(src_path))
        new_ext = src_ext if ext is None else ext
        new_path = os.path.join(arg_path, prefix + src_base + post_fix + new_ext)
    dirname = os.path.dirname(new_path)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    return new_path


def display_result(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2
    for i in range(detections.shape[0]):
        ymin, xmin, ymax, xmax = detections[i, :4].astype(int)
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k * 2])
                kp_y = int(detections[i, 4 + k * 2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)

        # 读取 score
        score = detections[i, -1]
        # 写 score 到图上（可选）
        cv2.putText(img, f"{score:.2f}", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


def yuv420_to_rgb(data_bytes, width, height):
    """
    data_bytes: bytes 或 bytearray，YUV420 (I420) 格式
    width, height: 图像尺寸
    返回: BGR numpy 数组 (H, W, 3)
    """
    yuv = np.frombuffer(data_bytes, dtype=np.uint8).reshape((height * 3 // 2, width))
    # rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV21)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    return rgb


def image_crop_ellipse(rgb_img, image_crop_config=None):
    """
    在原始 RGB 图像上裁切椭圆区域，并输出紧贴椭圆边缘的最小外接矩形区域。

    参数：
        rgb_img: 输入 RGB 图像
        center: (x, y)，椭圆中心，默认图像中心
        axes: (a, b)，椭圆半轴长度，默认覆盖整个图像

    返回：
        cropped_img: 紧贴椭圆外框的 RGB 图像
    """
    center = ()
    axes = ()
    height, width = rgb_img.shape[:2]

    if image_crop_config is None:
        # 默认中心为图像中心
        center = (width // 2, height // 2)
        # 默认椭圆半轴覆盖整个图像
        axes = (width // 2, height // 2)
    else:
        center = (image_crop_config["center_x"], image_crop_config["center_y"])
        axes = (image_crop_config["axes_w"], image_crop_config["axes_h"])

    # 创建二值 mask（椭圆区域为255）
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # 椭圆外设为黑色
    masked_img = cv2.bitwise_and(rgb_img, rgb_img, mask=mask)

    # 获取非零区域的最小外接矩形
    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪出紧贴椭圆边缘的图像区域
    cropped_img = masked_img[y:y + h, x:x + w]

    return cropped_img


def get_remove_distortion_mapping_matrix(camera_distortion_removal_parameter):
    w = camera_distortion_removal_parameter["image_axes_w"]
    h = camera_distortion_removal_parameter["image_axes_h"]
    fx = camera_distortion_removal_parameter["fx"]
    fy = camera_distortion_removal_parameter["fy"]
    cx = camera_distortion_removal_parameter["cx"]
    cy = camera_distortion_removal_parameter["cy"]

    k1 = camera_distortion_removal_parameter["k1"]
    k2 = camera_distortion_removal_parameter["k2"]
    p1 = camera_distortion_removal_parameter["p1"]
    p2 = camera_distortion_removal_parameter["p2"]
    k3 = camera_distortion_removal_parameter["k3"]
    k4 = camera_distortion_removal_parameter["k4"]
    k5 = camera_distortion_removal_parameter["k5"]
    k6 = camera_distortion_removal_parameter["k6"]

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)  # 内参矩阵
    D = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)  # 畸变系数
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)

    return map1, map2


def execute_camera_distortion_remove(frame, map1, map2):
    undistorted_img = cv2.remap(frame, map1, map2, cv2.INTER_CUBIC)
    return undistorted_img


def filter_detections(detections, box_min_rate_thread=0.15, max_widths=120, max_heights=105):
    # 过滤 bbox
    if box_min_rate_thread is not None:
        widths = detections[:, 2] - detections[:, 0]
        heights = detections[:, 3] - detections[:, 1]

        keep_mask = np.ones(len(detections), dtype=bool)
        if box_min_rate_thread is not None:
            widths_rate = widths / max_widths
            heights_rate = heights / max_heights
            keep_mask &= (widths_rate >= box_min_rate_thread) & (heights >= heights_rate)
        # if max_size is not None:
        #     keep_mask &= (widths <= max_size) & (heights <= max_size)

        detections = detections[keep_mask]
    return detections


class HandStateController:

    def __init__(self, threshold=2, time_thresh=0.01):
        self.threshold = threshold  # 连续帧阈值
        self.identification_number = math.ceil(threshold / 2)
        self.counter = 0  # 当前累计帧数
        self.event_state = False  # 当前状态输出

        # self.event_time_limit = time_thresh  #检测到手部的时间间隔
        # self.event_infer_send_time = time.time()  #上次发送检测到手部的
        # self.event_info_send_trigger = False  # 发送手部检测时间状态标志

    def update(self, detected: bool):
        """
        detected: bool, 当前帧是否检测到手
        返回: bool, 当前平滑后的状态
        """

        if detected:
            self.counter = min(self.counter + 1, self.threshold)  # 限制最大不超过 threshold
        else:
            self.counter = max(self.counter - 1, 0)  # 限制最小不低于 0

        if self.counter >= self.identification_number:
            self.event_state = True
        else:
            self.event_state = False

        # if self.event_state:
        #     current_time = time.time()
        #     delta_t = current_time - self.event_infer_send_time
        #     # print(delta_t)
        #     if delta_t > self.event_time_limit:
        #
        #         self.event_info_send_trigger = True
        #         self.event_infer_send_time = current_time
        #     else:
        #         self.event_info_send_trigger = False
        # else:
        #     self.event_info_send_trigger = False
