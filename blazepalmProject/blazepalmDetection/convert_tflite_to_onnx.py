import subprocess
import sys

import argparse
import subprocess
import sys

def convert_tflite_to_onnx(tflite_file="palm_detection_full.tflite", output_file="palm_detection_full.onnx",opset=11):
    cmd = [
        sys.executable,  # 使用当前 Python 解释器
        "-m", "tf2onnx.convert",
        "--opset", str(opset),
        "--tflite", tflite_file,
        "--output", output_file
    ]

    print("正在执行转换命令：")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 转换成功！ONNX 文件已生成：{output_file}")
    except subprocess.CalledProcessError as e:
        print("\n❌ 转换失败！")
        print("错误信息：", e)
    except FileNotFoundError:
        print("\n❌ 未找到 tf2onnx 模块，请先安装：")
        print("   pip install tf2onnx")
    return output_file

