from src.results import DetectionResults
import numpy as np
from abc import abstractmethod


class BasePreprocessing:
    @abstractmethod
    def __call__(
        self, image: np.ndarray, input_h: int, input_w: int, fill_value: int
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class BaseYolo:
    """Object detection model
    The __call__ method must accept `image` and return dict with keys:
    `output0` - output of YOLO model (tensor of shape [1, 4+num_classes, num_boxes])
    """

    @property
    @abstractmethod
    def input_h(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_w(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class BaseNonMaxSupression:
    @abstractmethod
    def __call__(
        self,
        output0: np.ndarray,
        max_output_boxes_per_class: int,
        iou_threshold: float,
        score_threshold: float,
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class BasePostprocessing:
    @abstractmethod
    def __call__(
        self, input_h: int, input_w: int, boxes_xywh: np.ndarray, padding_tlbr: np.ndarray
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError()


class BaseObjectDetector:
    def __init__(
        self,
        preprocessing: BasePreprocessing,
        yolo: BaseYolo,
        nms: BaseNonMaxSupression,
        postprocessing: BasePostprocessing,
    ):
        self.preprocessing = preprocessing
        self.yolo = yolo
        self.nms = nms
        self.postprocessing = postprocessing

    def __call__(
        self, image: np.ndarray, iou_threshold: float = 0.7, score_threshold: float = 0.25
    ):
        return self.inference(image, iou_threshold, score_threshold)

    def _preprocess(self, image: np.ndarray, fill_value: int = 114):
        return self.preprocessing(image, self.yolo.input_h, self.yolo.input_w, fill_value)

    def _detect(self, image: np.ndarray):
        return self.yolo(image)

    def _nms(
        self,
        output0: np.ndarray,
        max_output_boxes_per_class: int = 100,
        iou_threshold: float = 0.7,
        score_threshold: float = 0.25,
    ):
        return self.nms(output0, max_output_boxes_per_class, iou_threshold, score_threshold)

    def _postprocess(self, boxes_xywh: np.ndarray, padding_tlbr: np.ndarray):
        return self.postprocessing(self.yolo.input_h, self.yolo.input_w, boxes_xywh, padding_tlbr)

    def inference(self, image: np.ndarray, iou_threshold: float, score_threshold: float):
        preprocessed = self._preprocess(image)
        preprocessed_img, padding_tlbr = (
            preprocessed["preprocessed_img"],
            preprocessed["padding_tlbr"],
        )
        # add batch dim
        preprocessed_img = np.expand_dims(preprocessed_img, 0).astype(np.float32)
        yolo_output = self._detect(preprocessed_img)
        output0 = yolo_output["output0"]
        selected_output = self._nms(
            output0=output0,
            max_output_boxes_per_class=100,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        boxes_xywh, class_scores, class_ids = (
            selected_output["selected_boxes_xywh"],
            selected_output["selected_class_scores"],
            selected_output["selected_class_ids"],
        )
        postprocessed = self._postprocess(boxes_xywh, padding_tlbr)
        boxes_xywhn = postprocessed["boxes_xywhn"]
        if len(class_scores) == 0:
            return DetectionResults(image)
        return DetectionResults(image, boxes_xywhn, class_ids, class_scores)
