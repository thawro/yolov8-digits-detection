from ultralytics import YOLO
import torchvision

if __name__ == "__main__":
    # model = YOLO("yolov8n.pt")
    IMGSZ = 256
    # model.train(data="yolo_HWD+.yaml", epochs=50, imgsz=IMGSZ, device=0, batch=128)
    model = YOLO(
        "/home/shate/Desktop/projects/yolov8-digits-detection/runs/detect/train3/weights/best.pt"
    )
    model.export(format="onnx", imgsz=IMGSZ)
