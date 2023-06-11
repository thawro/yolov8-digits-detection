from src.transforms import xywh2xyxy
from src.utils.vision import non_maximum_supression
from src.visualization import plot_yolo_labels, ID2NAME

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

    def filter_by_idxs(self, idxs):
        self.xywhn = self.xywhn[idxs]
        self.cls = self.cls[idxs]
        self.conf = self.conf[idxs]


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
        return self.boxes.cls.tolist()

    @property
    def conf(self):
        return self.boxes.conf.tolist()

    def non_maximum_supression(self, iou_threshold):
        idxs = non_maximum_supression(self.boxes.xyxyn, self.boxes.conf, iou_threshold)
        self.boxes.filter_by_idxs(idxs)

    def visualize(self, plot=True):
        if self.is_empty:
            print("No objects detected")
            return self.orig_image
        bbox_img = plot_yolo_labels(
            image=self.orig_image,
            bboxes_xywhn=self.boxes.xywhn,
            class_ids=self.boxes.cls.tolist(),
            confidences=self.boxes.conf.tolist(),
            id2name=ID2NAME,
            # id2color=ID2COLOR,
        )
        if plot:
            plt.figure(figsize=(12, 12))
            plt.axis("off")
            plt.imshow(bbox_img)
        return bbox_img
