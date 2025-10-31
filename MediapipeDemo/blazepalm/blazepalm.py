import sys
import time

import ailia
import cv2
import numpy as np

import blazepalm_utils as but


# logger
from logging import getLogger  # noqa: E402

import util.webcamera_utils  # noqa: E402
from util.image_utils import imread, load_image  # noqa: E402
from util.model_utils import check_and_download_models  # noqa: E402
from util.arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from util.webcamera_utils import get_capture, get_writer  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'person_with_hands.jpg'
SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazePalm, on-device real-time palm detection.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)

args = update_parser(parser)


# ======================
# Parameters 2
# ======================
MODEL_NAME = 'palm_detection_full'


#Download palm_detection_full.tflite
#https://github.com/google/mediapipe/tree/master/mediapipe/modules/palm_detection
#python3 -m tf2onnx.convert --opset 11 --tflite palm_detection_full.tflite --output palm_detection_full.onnx
WEIGHT_PATH = f'palm_detection_full.onnx'

IMAGE_HEIGHT = 192
IMAGE_WIDTH = 192
ANCHOR_PATH = 'anchors_192.npy'
CHANNEL_FIRST = False



# ======================
# Utils
# ======================
def display_result(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]

        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1)

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize

    import onnxruntime
    net = onnxruntime.InferenceSession(WEIGHT_PATH)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = imread(image_path)
        img256, _, scale, pad = but.resize_pad(src_img[:, :, ::-1],IMAGE_WIDTH)
        input_data = img256.astype('float32') / 255.
        input_data = np.expand_dims(np.moveaxis(input_data, -1, 0), 0)
        if not CHANNEL_FIRST:
            input_data = input_data.transpose((0,2,3,1))

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for _ in range(5):
                start = int(round(time.time() * 1000))
                preds = net.predict([input_data])
                normalized_detections = but.postprocess(preds,anchor_path=ANCHOR_PATH,resolution=IMAGE_WIDTH)[0]
                detections = but.denormalize_detections(
                    normalized_detections, scale, pad, resolution=IMAGE_WIDTH
                )
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            
            input_name = net.get_inputs()[0].name
            preds = net.run(None, {input_name: input_data.astype(np.float32)})


        # postprocessing
        display_result(src_img, detections)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')


def main():

    recognize_from_image()


if __name__ == '__main__':
    main()
