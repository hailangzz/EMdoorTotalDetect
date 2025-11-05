import time
import os
import cv2
import datetime
import shutil

class ProjectDebug:

    def __init__(self,):
        self.debug_info = {"image_debug":{"save_frame_rgb_image_dir":"./rgb_images",
                        "interval":3,
                        "frame_count":0,
                        "save_image_number":0,
                        },
                      "log_debug":{"save_log_dir":"./logs/debug_logs.log",
                        "save_logs_number":0,
                        }
                      }


        self.save_image_dir = self.debug_info["image_debug"]['save_frame_rgb_image_dir']
        if os.path.exists(self.save_image_dir):
            shutil.rmtree(self.save_image_dir)  # 删除整个目录
        os.makedirs(self.save_image_dir)
        # os.makedirs(self.save_image_dir, exist_ok=True)

        self.save_log_dir = self.debug_info["log_debug"]['save_log_dir']
        os.makedirs(os.path.dirname(self.save_log_dir), exist_ok=True)
        #清空图片存储目录

        # 以追加模式打开日志文件
        self.file = open(self.save_log_dir , 'w', encoding='utf-8')

    def __del__(self):
        """
        析构函数：释放文件句柄
        """
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
            print(f"日志文件 {self.save_log_dir} 已关闭。")

    def save_frame_rgb_image(self,rgb_frame):        # 保存间隔（秒）


        self.debug_info["image_debug"]["frame_count"]+=1

        # if self.debug_info["image_debug"]["frame_count"]%10==0:

        # 图片文件名
        filename = os.path.join(self.save_image_dir, str(self.debug_info["image_debug"]["save_image_number"])+".png")
        # 保存图片
        # cv2.imwrite(filename, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(filename, rgb_frame)
        self.debug_info["image_debug"]["save_image_number"] += 1


    def save_detect_object_log(self,message: str):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.file.write(log_message)
        self.file.flush()  # 立即写入文件


# 使用示例
if __name__ == "__main__":
    logger = ProjectDebug()
    logger.save_detect_object_log("程序启动")

