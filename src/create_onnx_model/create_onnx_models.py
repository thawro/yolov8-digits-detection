from src.create_onnx_model import (
    create_onnx_preprocessing,
    create_onnx_yolo,
    create_onnx_NMS,
    create_onnx_postprocessing,
)
from src.utils.utils import MODELS_PATH


def create_models(
    yolo_ckpt_path: str, yolo_imgsz: int = 256, opset: int = 18, dirpath: str = str(MODELS_PATH)
):
    preprocessing_path = f"{dirpath}/preprocessing.onnx"
    yolo_path = f"{dirpath}/yolo.onnx"
    nms_path = f"{dirpath}/nms.onnx"
    postprocessing_path = f"{dirpath}/postprocessing.onnx"

    create_onnx_preprocessing(filepath=preprocessing_path, opset=opset)
    create_onnx_yolo(ckpt_path=yolo_ckpt_path, filepath=yolo_path, imgsz=yolo_imgsz, opset=opset)
    create_onnx_NMS(filepath=nms_path, opset=opset)
    create_onnx_postprocessing(filepath=postprocessing_path, opset=opset)


if __name__ == "__main__":
    dirpath = str(MODELS_PATH)
    yolo_ckpt_path = f"{dirpath}/best.pt"
    yolo_imgsz = 256
    create_models(
        yolo_ckpt_path=yolo_ckpt_path,
        yolo_imgsz=yolo_imgsz,
        dirpath=dirpath,
        opset=18,
    )
