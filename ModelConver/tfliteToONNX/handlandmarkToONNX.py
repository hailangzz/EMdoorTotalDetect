# 换源安装模块 pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple/ tflite2onnx

#模型转换命令 tflite2onnx /home/chenkejing/Downloads/hand_landmark_full.tflite hand_landmark.onnx


from tflite2onnx import convert

tflite_path = '/home/chenkejing/Downloads/hand_landmark_full.tflite'
onnx_path = './hand_landmark_full.onnx'
convert(tflite_path, onnx_path)  



