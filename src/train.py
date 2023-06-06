from ultralytics import YOLO
from src.utils.utils import ROOT

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    IMGSZ = 256
    model.train(data="../svhn.yaml", epochs=50, imgsz=IMGSZ, device=0, batch=128)
    model.export(format="onnx", imgsz=IMGSZ)
