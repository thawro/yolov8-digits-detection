import numpy as np
from src.ops import resize_pad, non_maximum_supression, xywh2xyxy
from src.object_detection.base import (
    BasePreprocessing,
    BaseNonMaxSupression,
    BasePostprocessing,
    BaseObjectDetector,
)
from src.object_detection.with_onnx import OnnxYolo, YOLO_PATH
from pathlib import Path


class Preprocessing(BasePreprocessing):
    def __call__(self, image: np.ndarray, input_h: int, input_w: int, fill_value: int):
        processed_image = image[..., :3]
        processed_image, padding_tlbr = resize_pad(processed_image, input_h, input_w, fill_value)
        processed_image = processed_image.transpose(2, 0, 1) / 255.0
        return {"preprocessed_img": processed_image, "padding_tlbr": padding_tlbr}


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
        boxes_xywh -= [pad_left, pad_top, 0, 0]
        pad_y = pad_top + pad_bottom
        pad_x = pad_left + pad_right
        h = input_h - pad_y
        w = input_w - pad_x
        boxes_xywhn = boxes_xywh / [w, h, w, h]
        return {"boxes_xywhn": boxes_xywhn}


class ObjectDetector(BaseObjectDetector):
    def __init__(self, yolo_path: str | Path = YOLO_PATH):
        super().__init__(
            preprocessing=Preprocessing(),
            yolo=OnnxYolo(yolo_path),
            nms=NonMaxSupression(),
            postprocessing=Postprocessing(),
        )
