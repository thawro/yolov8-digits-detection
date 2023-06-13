from ultralytics import YOLO
from src.utils.utils import MODELS_PATH
import os


def create_onnx_yolo(ckpt_path: str = "best.pt", filepath="yolo.onnx", imgsz: int = 256, opset=18):
    model = YOLO(ckpt_path)
    model.export(format="onnx", imgsz=imgsz, opset=opset)
    onnx_model_path = ckpt_path.replace(".pt", ".onnx")
    os.rename(onnx_model_path, filepath)


if __name__ == "__main__":
    ckpt_path = str(MODELS_PATH / "best.pt")
    filepath = str(MODELS_PATH / "yolo.onnx")
    create_onnx_yolo(ckpt_path, filepath, imgsz=256, opset=18)
