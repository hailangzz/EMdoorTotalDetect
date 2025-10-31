import onnx
from onnxsim import simplify

# 加载原始模型
model = onnx.load("/home/chenkejing/EMdoor_TotalProgram/hand_landmark_full.onnx")

# 简化模型
simplified_model, check = simplify(model)
if check:
    onnx.save(simplified_model, "./simplified_model.onnx")
else:
    print("简化失败，可能包含不支持的算子")
