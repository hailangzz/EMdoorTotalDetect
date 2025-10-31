import onnx
from onnx import numpy_helper
import numpy as np
import os

origin_model_path = r"/home/chenkejing/EMdoor_TotalProgram/ModelConver/TwoPartONNXcombin/MediaPipe-Hand-Detection_HandLandmarkDetector"


# 1. 载入模型定义（无参数）
model = onnx.load(os.path.join(origin_model_path,'model.onnx'))

# 2. 从 model.data 读取参数
#    这一步需要你清楚 model.data 格式，
#    假设是 numpy 的 .npy 格式（常见）
params = np.load(os.path.join(origin_model_path,'model.data'), allow_pickle=True).item()

# # 3. 把权重写回模型的初始化段
# #    假设 params 是 dict: {param_name: numpy_array}
# initializers = []
# for name, array in params.items():
#     tensor = numpy_helper.from_array(array, name)
#     initializers.append(tensor)

# # 覆盖现有 initializers
# model.graph.initializer.clear()
# model.graph.initializer.extend(initializers)

# # 4. 保存合并后的 onnx 文件
# onnx.save(model, os.path.join(origin_model_path,'model_merged.onnx'))
