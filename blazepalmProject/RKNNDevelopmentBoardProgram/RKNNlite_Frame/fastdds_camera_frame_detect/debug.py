import time
import os
import cv2


debug_values = {"save_frame_rgb_image_dir":"./rgb_images",
                "interval":3,
                "frame_count":0,
                "save_image_number":0}

def save_frame_rgb_image(rgb_frame):
    # 保存间隔（秒）
    save_dir = debug_values['save_frame_rgb_image_dir']
    os.makedirs(save_dir, exist_ok=True)

    debug_values["frame_count"]+=1

    if debug_values["frame_count"]%10==0:

        # 图片文件名
        filename = os.path.join(save_dir, str(debug_values["save_image_number"])+".png")
        # 保存图片
        cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        debug_values["save_image_number"] += 1

