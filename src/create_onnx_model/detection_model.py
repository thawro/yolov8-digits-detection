from ultralytics import YOLO
from src.utils.utils import ROOT


def create_onnx_detection(yolo_filepath: str = "models/best.pt", imgsz: int = 256):
    model = YOLO(yolo_filepath)
    model.export(format="onnx", imgsz=imgsz)


if __name__ == "__main__":
    IMGSZ = 256
    yolo_filepath = str(ROOT / "models/best.pt")
    create_onnx_detection(yolo_filepath, IMGSZ)
