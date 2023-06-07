from src.transforms import xywh2xyxy
from src.utils.vision import non_maximum_supression
from src.visualization import plot_yolo_labels, ID2NAME

import matplotlib.pyplot as plt
import numpy as np


class Boxes:
    def __init__(self, xywh, cls, conf, w, h):
        self.xywh = xywh
        self.w = w
        self.h = h
        self.cls = cls
        self.conf = conf

    @property
    def xyxyn(self):
        return xywh2xyxy(self.xywhn)

    @property
    def xywhn(self):
        bbox_divider = np.array([self.w, self.h, self.w, self.h])
        xywhn = np.divide(self.xywh, bbox_divider)
        return xywhn

    @property
    def xyxy(self):
        return xywh2xyxy(self.xywh)

    def filter_by_idxs(self, idxs):
        self.xywh = self.xywh[idxs]
        self.cls = self.cls[idxs]
        self.conf = self.conf[idxs]


class DetectionResults:
    def __init__(
        self,
        orig_image: np.ndarray | None = None,
        xywh: np.ndarray | None = None,
        cls: np.ndarray | None = None,
        conf: np.ndarray | None = None,
        w: int | None = None,
        h: int | None = None,
    ):
        self.orig_image = orig_image
        self.boxes = Boxes(xywh, cls, conf, w, h)

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
        idxs = non_maximum_supression(self.boxes.xyxy, self.boxes.conf, iou_threshold)
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
