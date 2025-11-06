from rknn.api import RKNN

def convert_model(tflite_file, rknn_file, is_image_model=False):
    rknn = RKNN()

    # 1. 配置参数
    if is_image_model:
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform='rk3588',
            quantized_dtype='w8a8'
        )
    else:
        rknn.config(
            target_platform='rk3588',
            quantized_dtype='w8a8'
        )

    # 2. 加载模型
    rknn.load_tflite(model=tflite_file)

    # 3. 构建（是否量化）
    if is_image_model:
        # 需要准备 dataset.txt
        rknn.build(do_quantization=True, dataset='dataset.txt')
    else:
        rknn.build(do_quantization=False)

    # 4. 导出模型
    rknn.export_rknn(rknn_file)
    rknn.release()

# 调用示例
convert_model('/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/hand_landmark_full.tflite', '/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/hand_landmark_full.rknn', False)
convert_model('/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/palm_detection_full.tflite', '/home/chenkejing/EMdoor_TotalProgram/MediapipeDemo/models/palm_detection_full.rknn', False)
# convert_model('gesture_embedder.tflite', 'gesture_embedder.rknn', False)
# convert_model('canned_gesture_classifier.tflite', 'canned_gesture_classifier.rknn', False)