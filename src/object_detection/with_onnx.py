import numpy as np
import onnxruntime as ort
from src.object_detection.base import BaseObjectDetector
from src.utils.utils import MODELS_PATH
from pathlib import Path

PREPROCESSING_PATH = MODELS_PATH / "preprocessing.onnx"
YOLO_PATH = MODELS_PATH / "yolo.onnx"
NMS_PATH = MODELS_PATH / "nms.onnx"
POSTPROCESSING_PATH = MODELS_PATH / "postprocessing.onnx"


def _const(value, dtype):
    return np.array([value], dtype=dtype)


class OnnxModel:
    """Base class for ONNX models loading"""

    def __init__(
        self,
        path: str | Path,
        providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        path = path if isinstance(path, str) else str(path)
        self.session = ort.InferenceSession(path, providers=providers)
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()
        self.inputs = [{"name": input.name, "shape": input.shape} for input in model_inputs]
        self.outputs = [{"name": output.name, "shape": output.shape} for output in model_outputs]

    def __call__(self, inputs: dict):
        output_names = [output["name"] for output in self.outputs]
        outputs = self.session.run(output_names, inputs)
        return {name: output for name, output in zip(output_names, outputs)}


class OnnxPreprocessing(OnnxModel):
    """Preprocessing ONNX model
    The __call__ method must accept `image`, `input_h`, `input_w` and `fill_value`
    and return dict with keys:
    `preprocessed_img` - image after preprocessing (resize -> pad -> normalize)
    `padding_tlbr` - padding added to top, left, bottom and right (needed to postprocess boxes)
    """

    def __init__(self, path: str | Path):
        super().__init__(path, providers=["CPUExecutionProvider"])

    def __call__(self, image: np.ndarray, input_h: int, input_w: int, fill_value: int):
        inputs = {
            "image": image,
            "input_h": _const(input_h, np.int32),
            "input_w": _const(input_w, np.int32),
            "fill_value": _const(fill_value, np.uint8),
        }
        return super().__call__(inputs)


class OnnxYolo(OnnxModel):
    """Object detection ONNX model
    The __call__ method must accept `image` and return dict with keys:
    `output0` - output of YOLO model (tensor of shape [1, 4+num_classes, num_boxes])
    """

    def __init__(
        self, path: str | Path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    ):
        super().__init__(path, providers)
        image_input = self.inputs[0]
        self.input_name = image_input["name"]
        _, self.input_c, self.input_h, self.input_w = image_input["shape"]

    def __call__(self, image):
        inputs = {"images": image}
        return super().__call__(inputs)


class OnnxNonMaxSupression(OnnxModel):
    """NonMaxSupression ONNX model
    The __call__ method must accept `output0`, `max_output_boxes_per_class`,
    `iou_threshold`, `score_threshold` and return dict with keys:
    `selected_boxes_xywh` - xywh boxes left after NMS filtering
    `selected_class_scores` - scores of the boxes
    `selected_class_ids` - class_ids of the boxes
    """

    def __init__(self, path: str | Path):
        super().__init__(path, providers=["CPUExecutionProvider"])

    def __call__(
        self,
        output0: np.ndarray,
        max_output_boxes_per_class: int = 100,
        iou_threshold: float = 0.7,
        score_threshold: float = 0.25,
    ):
        inputs = {
            "output0": output0,
            "max_output_boxes_per_class": _const(max_output_boxes_per_class, np.int32),
            "iou_threshold": _const(iou_threshold, np.float32),
            "score_threshold": _const(score_threshold, np.float32),
        }
        return super().__call__(inputs)


class OnnxPostprocessing(OnnxModel):
    """Postprocessing ONNX model
    The __call__ method must accept `input_h`, `input_w`, `boxes_xywh` and `padding_tlbr`
    and return dict with keys:
    `boxes_xywhn` - Postprocessed boxes (normalized wrt original image)
    """

    def __init__(self, path: str | Path):
        super().__init__(path, providers=["CPUExecutionProvider"])

    def __call__(
        self, input_h: int, input_w: int, boxes_xywh: np.ndarray, padding_tlbr: np.ndarray
    ):
        inputs = {
            "boxes_xywh": boxes_xywh,
            "input_h": _const(input_h, np.int32),
            "input_w": _const(input_w, np.int32),
            "padding_tlbr": padding_tlbr,
        }
        return super().__call__(inputs)


class OnnxObjectDetector(BaseObjectDetector):
    def __init__(
        self,
        preprocessing_path: str | Path = PREPROCESSING_PATH,
        yolo_path: str | Path = YOLO_PATH,
        nms_path: str | Path = NMS_PATH,
        postprocessing_path: str | Path = POSTPROCESSING_PATH,
    ):
        super().__init__(
            preprocessing=OnnxPreprocessing(preprocessing_path),
            yolo=OnnxYolo(yolo_path),
            nms=OnnxNonMaxSupression(nms_path),
            postprocessing=OnnxPostprocessing(postprocessing_path),
        )
