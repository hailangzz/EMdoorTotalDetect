import os
from ultralytics import YOLO
import cv2

def run_inference(model_path, imgs_dir, save_dir):
    # 加载模型
    model = YOLO(model_path)

    # 结果保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 支持的图片格式
    exts = ('.jpg', '.jpeg', '.png', '.bmp')

    # 遍历目录下所有图片
    for img_name in os.listdir(imgs_dir):
        if not img_name.lower().endswith(exts):
            continue

        img_path = os.path.join(imgs_dir, img_name)
        print(f"Processing {img_path}")

        # 推理
        results = model(img_path)[0]

        # 读取原图
        img = cv2.imread(img_path)

        # 绘制检测框
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls)
            conf = float(box.conf)
            label = f"{model.names[cls]} {conf:.2f}"

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # 保存结果
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)
        print(f"> Saved: {save_path}")


if __name__ == "__main__":
    model_path = "/home/chenkejing/PycharmProjects/ultralytics/runs/detect/train3/weights/best.pt"         # 修改为你的模型路径
    imgs_dir = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/wire_images_test"             # 输入图片目录
    save_dir = "./results"            # 结果输出目录

    run_inference(model_path, imgs_dir, save_dir)
