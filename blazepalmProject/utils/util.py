import os
import cv2

def read_image_from_video(video_path="../database/HandsDance.mp4"):
    save_frames = False  # 如果你想保存帧图像，改为 True

    cap = cv2.VideoCapture(video_path)  # 打开默认摄像头
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        exit()

    frame_count = 0
    print("开始读取视频帧...")

    # === 循环读取每一帧 ===
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束或出错。")
            break

        frame_count += 1
        print(f"📸 当前帧编号: {frame_count}")

        # === 显示当前帧 ===
        cv2.imshow("Video Frame", frame)

        # === 可选：保存每一帧到本地 ===
        if save_frames:
            cv2.imwrite(f"frame_{frame_count:04d}.jpg", frame)

        # === 按 'q' 退出播放 ===
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === 释放资源 ===
    cap.release()
    cv2.destroyAllWindows()


read_image_from_video(video_path="../database/HandsDance.mp4")