from ultralytics import YOLO


def train(
    ckpt_path: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: tuple[int, int] = (256, 256),
    batch: int = 128,
    export_format: str | None = "onnx",
):
    model = YOLO(ckpt_path)
    model.train(data="yolo_HWD+.yaml", epochs=epochs, imgsz=imgsz, device=0, batch=batch)
    if export_format is not None:
        model.export(format=export_format, imgsz=imgsz)


if __name__ == "__main__":
    IMGSZ = 256
    CKPT_PATH = "yolov8n.pt"
    EPOCHS = 50
    IMGSZ = (256, 256)
    BATCH = 128
    EXPORT_FORMAT = "onnx"
    train(ckpt_path=CKPT_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, export_format=EXPORT_FORMAT)
