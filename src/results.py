from src.ops import xywh2xyxy
from src.visualization import plot_yolo_labels
from src.utils.utils import ID2NAME

import matplotlib.pyplot as plt
import numpy as np


class Boxes:
    def __init__(self, xywhn, cls, conf, h, w):
        self.xywhn = xywhn
        self.h = h
        self.w = w
        self.cls = cls
        self.conf = conf

    @property
    def xyxyn(self):
        return xywh2xyxy(self.xywhn)

    @property
    def xywh(self):
        bbox_multiplier = np.array([self.w, self.h, self.w, self.h])
        return np.multiply(self.xywhn, bbox_multiplier)

    @property
    def xyxy(self):
        return xywh2xyxy(self.xywh)


class DetectionResults:
    def __init__(
        self,
        orig_image: np.ndarray,
        xywhn: np.ndarray | None = None,
        cls: np.ndarray | None = None,
        conf: np.ndarray | None = None,
    ):
        self.orig_image = orig_image
        h, w = orig_image.shape[:2]
        self.boxes = Boxes(xywhn, cls, conf, h, w)

    @property
    def is_empty(self):
        return self.boxes.cls is None

    @property
    def class_ids(self):
        return self.boxes.cls

    @property
    def conf(self):
        return self.boxes.conf

    def visualize(self, plot=False):
        if self.is_empty:
            print("No objects detected")
            return self.orig_image
        bbox_img = plot_yolo_labels(
            image=self.orig_image,
            bboxes_xywhn=self.boxes.xywhn,
            class_ids=self.class_ids,
            confidences=self.conf,
            id2name=ID2NAME,
        )
        if plot:
            plt.figure(figsize=(12, 12))
            plt.axis("off")
            plt.imshow(bbox_img)
        return bbox_img
