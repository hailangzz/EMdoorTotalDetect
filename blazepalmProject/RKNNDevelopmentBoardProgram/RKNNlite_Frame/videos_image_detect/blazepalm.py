import cv2
import time
import blazepalm_utils as but
import RKNNInference


if __name__ == "__main__":

    BlazePalmModelInfo = {"IMAGE_HEIGHT": 192,
                          "IMAGE_WIDTH": 192,
                          "CHANNEL_FIRST": False,
                          "RKNN_MODEL": "palm_detection_full.rknn",
                          "ANCHOR_PATH": "anchors_192.npy"
                          }
    save_dir = "./camera_images"  # 保存路径
    videos_path = r"./HandsDance.mp4"

    rknn_infer = RKNNInference.RKNNInference(BlazePalmModelInfo)

    try:
        rknn_infer.open_camera(videos_path)  # 打开主摄像头
        for i in range(300):  # 获取 10 帧
            outputs, frame = rknn_infer.infer()
            print(f"number {i + 1} frame infer succeed!")
            for object_index, out in enumerate(outputs):
                print(f"Output[{object_index}] shape: {out.shape}")


            normalized_detections = but.postprocess(outputs, anchor_path=BlazePalmModelInfo["ANCHOR_PATH"],
                                                    resolution=BlazePalmModelInfo["IMAGE_WIDTH"])[0]
            detections = but.denormalize_detections(normalized_detections,
                                                    rknn_infer.image_resize_pad_info["scale"],
                                                    rknn_infer.image_resize_pad_info["pad"],
                                                    resolution=BlazePalmModelInfo["IMAGE_WIDTH"])

            # ======================
            # 显示与保存结果
            # ======================
            result_img = but.display_result(frame, detections) # 添加检测框，到图像数据中


            savepath = but.get_savepath(f"image_{i + 1}.jpg", save_dir)  #图像存储名称
            cv2.imwrite(savepath, result_img)
            print(f'infer result save path: {savepath}')

            time.sleep(3)  # 间隔

    finally:
        rknn_infer.release()


