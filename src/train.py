from ultralytics import YOLO
from src.utils.utils import ROOT


def export_model(ckpt_path: str, imgsz: tuple[int, int] = (256, 256), format: str = "onnx"):
    model = YOLO(ckpt_path)
    model.export(format=format, imgsz=imgsz)


def train_model(
    ckpt_path: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: tuple[int, int] = (256, 256),
    batch: int = 128,
    train: bool = True,
    export: bool = True,
):
    if train:
        model = YOLO(ckpt_path)
        model.train(data="yolo_HWD+.yaml", epochs=epochs, imgsz=imgsz, device=0, batch=batch)
    if export:
        export_model(ckpt_path, imgsz)


if __name__ == "__main__":
    CKPT_PATH = "yolov8n.pt"
    # CKPT_PATH = ROOT / "runs/detect/train4/weights/best.pt"

    EPOCHS = 100
    IMGSZ = (256, 256)
    BATCH = 128
    train_model(
        ckpt_path=CKPT_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, train=False, export=True
    )
