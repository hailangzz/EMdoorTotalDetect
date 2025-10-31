以下是一个完整的 tflite 转 onnx 的 Python 脚本：

```python
#!/usr/bin/env python3
"""
TensorFlow Lite 转 ONNX 格式转换脚本
支持自动检测模型输入输出规格
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("请先安装 ONNX 相关依赖：pip install onnx onnxruntime")
    sys.exit(1)

def install_tflite2onnx():
    """安装 tflite2onnx 包"""
    try:
        import tflite2onnx
        return True
    except ImportError:
        print("正在安装 tflite2onnx...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tflite2onnx"])
            import tflite2onnx
            print("tflite2onnx 安装成功！")
            return True
        except Exception as e:
            print(f"安装失败: {e}")
            return False

def get_tflite_model_info(tflite_path):
    """获取 tflite 模型的输入输出信息"""
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("=" * 50)
        print("TFLite 模型信息:")
        print("=" * 50)
        
        print("\n输入张量:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}:")
            print(f"    名称: {detail.get('name', 'Unknown')}")
            print(f"    形状: {detail['shape']}")
            print(f"    数据类型: {detail['dtype']}")
            print(f"    量化信息: {detail.get('quantization', '无')}")
        
        print("\n输出张量:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}:")
            print(f"    名称: {detail.get('name', 'Unknown')}")
            print(f"    形状: {detail['shape']}")
            print(f"    数据类型: {detail['dtype']}")
            print(f"    量化信息: {detail.get('quantization', '无')}")
        
        return input_details, output_details
    except Exception as e:
        print(f"获取模型信息失败: {e}")
        return None, None

def convert_tflite_to_onnx(tflite_path, onnx_path, input_shapes=None, output_names=None):
    """转换 TFLite 模型到 ONNX 格式"""
    
    # 检查输入文件
    if not os.path.exists(tflite_path):
        print(f"错误: 输入文件 {tflite_path} 不存在")
        return False
    
    # 安装并导入 tflite2onnx
    if not install_tflite2onnx():
        return False
    
    from tflite2onnx import convert
    
    try:
        # 构建转换命令参数
        cmd_args = [tflite_path, onnx_path]
        
        # 添加输入形状参数
        if input_shapes:
            for shape in input_shapes:
                cmd_args.extend(['--input-shape', shape])
        
        # 添加输出名称参数
        if output_names:
            for name in output_names:
                cmd_args.extend(['--output-name', name])
        
        print(f"\n开始转换: {tflite_path} -> {onnx_path}")
        
        # 执行转换
        convert(cmd_args)
        
        print("转换完成！")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def verify_onnx_model(onnx_path, test_inputs=None):
    """验证转换后的 ONNX 模型"""
    try:
        # 加载并检查 ONNX 模型
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX 模型格式验证通过")
        
        # 创建推理会话
        session = ort.InferenceSession(onnx_path)
        
        # 打印模型信息
        print("\nONNX 模型信息:")
        print("=" * 30)
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print("输入:")
        for inp in inputs:
            print(f"  名称: {inp.name}, 形状: {inp.shape}, 类型: {inp.type}")
        
        print("输出:")
        for out in outputs:
            print(f"  名称: {out.name}, 形状: {out.shape}, 类型: {out.type}")
        
        # 如果提供了测试输入，进行推理测试
        if test_inputs:
            print("\n进行推理测试...")
            try:
                # 准备输入数据
                feed_dict = {}
                for inp in inputs:
                    if inp.name in test_inputs:
                        feed_dict[inp.name] = test_inputs[inp.name]
                    else:
                        # 创建随机测试数据
                        shape = [dim if dim > 0 else 1 for dim in inp.shape]
                        feed_dict[inp.name] = np.random.random(shape).astype(np.float32)
                
                # 运行推理
                results = session.run(None, feed_dict)
                
                print("✓ 推理测试成功！")
                for i, result in enumerate(results):
                    print(f"  输出 {i}: 形状 {result.shape}")
                    
            except Exception as e:
                print(f"⚠ 推理测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"ONNX 模型验证失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='TFLite 转 ONNX 转换工具')
    parser.add_argument('input', help='输入的 TFLite 模型文件路径')
    parser.add_argument('-o', '--output', help='输出的 ONNX 文件路径（可选）')
    parser.add_argument('--input-shape', nargs='+', help='输入形状，例如: 1,224,224,3')
    parser.add_argument('--output-name', nargs='+', help='输出名称')
    parser.add_argument('--no-verify', action='store_true', help='跳过模型验证')
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output:
        onnx_path = args.output
    else:
        input_path = Path(args.input)
        onnx_path = str(input_path.with_suffix('.onnx'))
    
    # 获取模型信息
    input_details, output_details = get_tflite_model_info(args.input)
    if not input_details:
        sys.exit(1)
    
    # 构建转换参数
    input_shapes = []
    output_names = []
    
    # 自动从模型信息中提取输入形状
    if not args.input_shape and input_details:
        for detail in input_details:
            shape_str = ','.join(map(str, detail['shape']))
            input_shapes.append(shape_str)
            print(f"自动检测到输入形状: {shape_str}")
    
    # 使用用户指定的输入形状
    if args.input_shape:
        input_shapes = args.input_shape
    
    # 使用用户指定的输出名称
    if args.output_name:
        output_names = args.output_name
    
    # 执行转换
    success = convert_tflite_to_onnx(
        args.input, 
        onnx_path, 
        input_shapes=input_shapes,
        output_names=output_names
    )
    
    if not success:
        print("转换失败！")
        sys.exit(1)
    
    # 验证模型
    if not args.no_verify:
        print("\n" + "=" * 50)
        print("验证转换结果...")
        print("=" * 50)
        
        # 准备测试输入数据
        test_inputs = {}
        if input_details:
            for detail in input_details:
                input_name = detail.get('name', f'input_{len(test_inputs)}')
                shape = [dim if dim > 0 else 1 for dim in detail['shape']]
                test_inputs[input_name] = np.random.random(shape).astype(np.float32)
        
        verify_onnx_model(onnx_path, test_inputs)
    
    print(f"\n转换完成！ONNX 模型保存至: {onnx_path}")

if __name__ == "__main__":
    main()


# ```

# ## 使用方法

# ### 1. 基本用法（自动检测）：
# ```bash
# python tflite_to_onnx.py hand_landmark.tflite
# ```

# ### 2. 指定输出路径：
# ```bash
# python tflite_to_onnx.py hand_landmark.tflite -o my_model.onnx
# ```

# ### 3. 手动指定输入形状：
# ```bash
# python tflite_to_onnx.py hand_landmark.tflite --input-shape 1,224,224,3
# ```

# ### 4. 跳过验证：
# ```bash
# python tflite_to_onnx.py hand_landmark.tflite --no-verify
# ```

# ## 依赖安装

# 运行前请确保安装所需依赖：

# ```bash
# pip install tensorflow onnx onnxruntime tflite2onnx
# ```

# ## 脚本功能

# 1. **自动检测**：自动读取 tflite 模型的输入输出规格
# 2. **灵活转换**：支持手动指定输入形状和输出名称
# 3. **验证检查**：自动验证转换后的 ONNX 模型格式和推理功能
# 4. **错误处理**：完善的错误处理和用户提示
# 5. **详细日志**：提供详细的转换过程和模型信息

# ## 使用示例

# 假设你有一个 MediaPipe Hand Landmark 模型：

# ```bash
# # 下载模型
# wget https://storage.googleapis.com/mediapipe-models/hand_landmark/hand_landmark.tflite

# # 转换为 ONNX
# python tflite_to_onnx.py hand_landmark.tflite -o hand_landmark.onnx
# ```

# 这个脚本会自动处理大多数常见的转换场景，并提供详细的调试信息。