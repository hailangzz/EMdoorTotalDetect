import json
from ultralytics import YOLO

# 加载你的模型
model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/runs/my_wire_exp/yolov8_focus_v1/weights/best.pt")
# model = YOLO("/home/chenkejing/PycharmProjects/ultralytics/runs/detect/train8/weights/best.pt")

# 评估模型
metrics = model.val(data="wire_test.yaml")
print(metrics)
# 转成 Python dict（metrics 是 ultralytics 对象，需要转换）
results_dict = {
    "precision": float(metrics.box.mp),         # P
    "recall": float(metrics.box.mr),            # R
    "mAP50": float(metrics.box.map50),          # mAP@50
    "mAP50_95": float(metrics.box.map),         # mAP@50-95
}
print(results_dict)
# 保存到本地 JSON 文件
save_path = "eval_wire_result_model8_foucs.json"
with open(save_path, "w") as f:
    json.dump(results_dict, f, indent=4)

print(f"测试结果已保存到：{save_path}")
print(results_dict)
