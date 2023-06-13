import numpy as np
from src.ops import resize_pad, non_maximum_supression, xywh2xyxy
from src.object_detection.base import (
    BasePreprocessing,
    BaseNonMaxSupression,
    BasePostprocessing,
    BaseObjectDetector,
)
from src.object_detection.with_onnx import OnnxYolo


class Preprocessing(BasePreprocessing):
    def __call__(self, image: np.ndarray, input_h: int, input_w: int, fill_value: int):
        image = image[..., :3]
        image, padding_tlbr = resize_pad(image, input_h, input_w, fill_value)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        # input_tensor = np.expand_dims(image, 0).astype(np.float32)
        return {"preprocessed_img": image, "padding_tlbr": padding_tlbr}


class NonMaxSupression(BaseNonMaxSupression):
    def __call__(self, output0, max_output_boxes_per_class, iou_threshold, score_threshold):
        predictions = np.squeeze(output0).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        scores_mask = scores > score_threshold

        predictions = predictions[scores_mask, :]
        scores = scores[scores_mask]

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes_xywh = predictions[:, :4]

        boxes_xyxy = xywh2xyxy(boxes_xywh)

        idxs = non_maximum_supression(boxes_xyxy, scores, iou_threshold)

        return {
            "selected_boxes_xywh": boxes_xywh[idxs],
            "selected_class_scores": scores[idxs],
            "selected_class_ids": class_ids[idxs],
        }


class Postprocessing(BasePostprocessing):
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


class ObjectDetector(BaseObjectDetector):
    def __init__(self, yolo_path):
        super().__init__(
            preprocessing=Preprocessing(),
            yolo=OnnxYolo(yolo_path),
            nms=NonMaxSupression(),
            postprocessing=Postprocessing(),
        )
