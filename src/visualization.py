import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.ops import xywhn2xywh, xywh2xyxy


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_bbox(
    image: np.ndarray,
    bbox_xyxy: np.ndarray | list[int] | tuple[int, int, int, int],
    class_name: str,
    confidence: float | None,
    color: tuple[int, int, int] = (255, 0, 0),
    txt_color: tuple[int, int, int] = (255, 255, 255),
    lw: float | None = None,
):
    """Visualizes a single bounding box on the image

    Args:
        image (np.ndarray): Image to plot box on.
        bbox_xyxy (np.ndarray | list[int] | tuple[int, int, int, int]):
            xmin, ymin, xmax, ymax box coordinates.
        class_name (str): Name of the object inside the box.
        confidence (float, optional): Confidence of the prediction.
        color (tuple[int, int, int], optional): Rectangle color. Defaults to (255, 0, 0).
        txt_color (tuple[int, int, int], optional): Text color. Defaults to (255, 255, 255).
        lw (float | None, optional): Rectangle line width. Defaults to None.

    Returns:
        _type_: _description_
    """
    lw = lw or max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    x_min, y_min, x_max, y_max = bbox_xyxy
    if confidence is not None:
        txt_label = f"{class_name} {confidence:.1f}"
    else:
        txt_label = class_name
    p1, p2 = (x_min, y_min), (x_max, y_max)
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(txt_label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        image,
        txt_label,
        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
        0,
        lw / 3,
        txt_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return image


def plot_yolo_labels(
    image: np.ndarray,
    bboxes_xywhn: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray | None = None,
    id2name: dict[int, str] | None = None,
    plot: bool = False,
) -> np.ndarray:
    """Plot predicted boxes and labels for an image.

    Args:
        image (np.ndarray): Image to plot box on.
        bboxes_xywhn (np.ndarray): Predicted boxes in xywhn format.
        class_ids (np.ndarray): Class id of each box.
        confidences (np.ndarray, optional): Prediction confidences of each box.
        id2name (dict[int, str], optional): Mapping from class_id to class_name.
            Defaults to None (ids are used as class names).
        plot (bool, optional): Whether to plot labels using matplotlib. Defaults to False.

    Returns:
        np.ndarray: Image with ploted boxes
    """
    if id2name is None:
        id2name = {class_id: str(class_id) for class_id in class_ids}
    boxes_img = image.copy()
    img_h, img_w, img_c = boxes_img.shape
    if not isinstance(bboxes_xywhn, np.ndarray):
        bboxes_xywhn = np.array(bboxes_xywhn)
    if confidences is None:
        confidences = [None] * len(class_ids)
    bboxes_xywh = xywhn2xywh(bboxes_xywhn, h=img_h, w=img_w)
    bboxes_xyxy = xywh2xyxy(bboxes_xywh).tolist()
    for bbox, class_id, conf in zip(bboxes_xyxy, class_ids, confidences):
        class_name = id2name[class_id]
        color = colors(class_id)
        boxes_img = plot_bbox(boxes_img, bbox, class_name, conf, color, txt_color=(0, 0, 0))
    if plot:
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.imshow(boxes_img)
        plt.show()
    return boxes_img
