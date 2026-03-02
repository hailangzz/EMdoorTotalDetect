import os
import argparse
import sys

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont

from salt.editor import Editor
from salt.interface import ApplicationInterface


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model-path", type=str, default="sam_onnx.onnx")
    parser.add_argument("--dataset-path", type=str, default="dataset")
    parser.add_argument("--categories", type=str)
    args = parser.parse_args()

    onnx_model_path = args.onnx_model_path
    dataset_path = args.dataset_path

    categories = args.categories.split(",") if args.categories else None
    coco_json_path = os.path.join(dataset_path, "annotations.json")

    editor = Editor(
        onnx_model_path,
        dataset_path,
        categories=categories,
        coco_json_path=coco_json_path
    )

    app = QApplication(sys.argv)

    # ⭐⭐⭐ 关键一行，解决 Linux 中文乱码 ⭐⭐⭐
    app.setFont(QFont("Noto Sans CJK SC"))

    window = ApplicationInterface(app, editor)
    window.show()

    sys.exit(app.exec_())
