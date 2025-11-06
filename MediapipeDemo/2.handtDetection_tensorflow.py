import cv2
import numpy as np
import tensorflow as tf

# ========== 1. 加载模型 ==========
model_path = "palm_detection_full.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded:", model_path)
print("Input shape:", input_details[0]['shape'])
print("Output shapes:", [d['shape'] for d in output_details])

# ========== 2. 预处理输入图像 ==========
def preprocess_image(image_path, input_shape):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[1], input_shape[2]
    img_resized = cv2.resize(img_rgb, (w, h))
    input_data = np.expand_dims(img_resized.astype(np.float32), axis=0)
    return img, input_data

image_path = "pointing_up.jpg"
original_img, input_data = preprocess_image(image_path, input_details[0]['shape'])

# ========== 3. 生成官方版本的 anchors ==========
def generate_palm_anchors(input_size=192):
    strides = [8, 16, 32, 32, 32]
    anchors = []
    for stride in strides:
        fmap_h = int(np.ceil(input_size / stride))
        fmap_w = int(np.ceil(input_size / stride))
        for y in range(fmap_h):
            for x in range(fmap_w):
                cx = (x + 0.5) / fmap_w
                cy = (y + 0.5) / fmap_h
                anchors.append([cx, cy])
    anchors = np.array(anchors, dtype=np.float32)
    print(f"✅ Anchors generated: {anchors.shape}")
    return anchors

input_h = input_details[0]['shape'][1]  # usually 192
anchors = generate_palm_anchors(input_size=input_h)

# ========== 4. 模型推理 ==========
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# output 0: regressors (位置 + 关键点)
# output 1: classificators (置信度)
raw_boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # (2016, 18)
raw_scores = interpreter.get_tensor(output_details[1]['index'])[0]  # (2016, 1)
scores = raw_scores.squeeze()

print(raw_scores)

# # ========== 5. 解码 anchors ==========
# def decode_boxes(raw_boxes, anchors):
#     """
#     raw_boxes: (N, 18)
#     anchors: (N, 2)
#     return: boxes (N, 4), keypoints (N, 7, 2)
#     """
#     x_center = raw_boxes[:, 0] / 192.0 + anchors[:, 0]
#     y_center = raw_boxes[:, 1] / 192.0 + anchors[:, 1]
#     w = raw_boxes[:, 2] / 192.0
#     h = raw_boxes[:, 3] / 192.0

#     x_min = x_center - w / 2
#     y_min = y_center - h / 2
#     x_max = x_center + w / 2
#     y_max = y_center + h / 2
#     boxes = np.stack([x_min, y_min, x_max, y_max], axis=-1)

#     # 手掌关键点解码（7个点，每个2维）
#     keypoints = np.zeros((raw_boxes.shape[0], 7, 2))
#     for i in range(7):
#         keypoints[:, i, 0] = raw_boxes[:, 4 + i * 2] / 192.0 + anchors[:, 0]
#         keypoints[:, i, 1] = raw_boxes[:, 4 + i * 2 + 1] / 192.0 + anchors[:, 1]

#     return boxes, keypoints

# decoded_boxes, decoded_kps = decode_boxes(raw_boxes, anchors)



# # ========== 6. NMS (非极大值抑制) ==========
# def compute_iou(box, boxes):
#     x_min = np.maximum(box[0], boxes[:, 0])
#     y_min = np.maximum(box[1], boxes[:, 1])
#     x_max = np.minimum(box[2], boxes[:, 2])
#     y_max = np.minimum(box[3], boxes[:, 3])
#     inter = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
#     area1 = (box[2] - box[0]) * (box[3] - box[1])
#     area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     return inter / (area1 + area2 - inter + 1e-5)

# def non_max_suppression(boxes, scores, iou_threshold=0.3, max_detections=5):
#     idxs = np.argsort(scores)[::-1]
#     selected = []
#     while len(idxs) > 0 and len(selected) < max_detections:
#         i = idxs[0]
#         selected.append(i)
#         if len(idxs) == 1:
#             break
#         iou = compute_iou(boxes[i], boxes[idxs[1:]])
#         idxs = idxs[1:][iou < iou_threshold]
#     return selected

# score_thresh = 0.7
# mask = scores > score_thresh
# filtered_boxes = decoded_boxes[mask]
# filtered_kps = decoded_kps[mask]
# filtered_scores = scores[mask]

# selected = non_max_suppression(filtered_boxes, filtered_scores)
# final_boxes = filtered_boxes[selected]
# final_kps = filtered_kps[selected]
# final_scores = filtered_scores[selected]

# print(f"✅ Detected {len(final_boxes)} palms after NMS.")

# # ========== 7. 可视化 ==========
# h, w, _ = original_img.shape
# for i, box in enumerate(final_boxes):
#     x_min = int(box[0] * w)
#     y_min = int(box[1] * h)
#     x_max = int(box[2] * w)
#     y_max = int(box[3] * h)
#     cv2.rectangle(original_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     cv2.putText(original_img, f"{final_scores[i]:.2f}",
#                 (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5, (0, 255, 0), 1)
#     # 绘制手掌关键点
#     for (px, py) in final_kps[i]:
#         cx = int(px * w)
#         cy = int(py * h)
#         cv2.circle(original_img, (cx, cy), 3, (0, 0, 255), -1)

# cv2.imshow("Palm Detection", original_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
