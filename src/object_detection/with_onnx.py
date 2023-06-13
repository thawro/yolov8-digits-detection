import numpy as np
import onnxruntime as ort
from src.object_detection.base import BaseObjectDetector


def _const(value, dtype):
    return np.array([value], dtype=dtype)


class OnnxModel:
    """Base class for ONNX models loading"""

    def __init__(self, path: str, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
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

    def __init__(self, path):
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

    def __init__(self, path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
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

    def __init__(self, path):
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


# TODO: remove
class Postprocessing:
    def __call__(
        self, input_h: int, input_w: int, boxes_xywh: np.ndarray, padding_tlbr: np.ndarray
    ):
        pad_top, pad_left, pad_bottom, pad_right = padding_tlbr
        boxes_xywh[:, 0] -= pad_left
        boxes_xywh[:, 1] -= pad_top
        h = input_h - (pad_top + pad_bottom)
        w = input_w - (pad_left + pad_right)
        boxes_xywhn = np.divide(boxes_xywh, np.array([w, h, w, h]))
        return {"boxes_xywhn": boxes_xywhn}


class OnnxObjectDetector(BaseObjectDetector):
    def __init__(self, preprocessing_path: str, yolo_path: str, nms_path: str):
        super().__init__(
            preprocessing=OnnxPreprocessing(preprocessing_path),
            yolo=OnnxYolo(yolo_path),
            nms=OnnxNonMaxSupression(nms_path),
            postprocessing=Postprocessing(),
        )
