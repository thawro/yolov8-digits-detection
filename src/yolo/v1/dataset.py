from typing import Literal, Optional, Callable, Any
from src.yolo.base.dataset import YOLOBaseDataset
import torch


class YOLOv1Dataset(YOLOBaseDataset):
    def __init__(
        self,
        S: int,
        C: int,
        B: int,
        root: str,
        split: Literal["train", "test", "extra"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.S = S  # grid size
        self.C = C  # num classes
        self.B = B  # num boxes
        super().__init__(root, split, transform, target_transform, download)

    def __getitem__(self, idx: int) -> Any:
        image, annots = self.get_raw_data(idx)
        # annotations cols:
        # class x_center y_center width height
        annots = torch.Tensor([[float(x) for x in label.split(" ")] for label in annots])

        bboxes = annots[:, 1:]
        labels = annots[:, 0]
        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=bboxes, labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
        bboxes, labels = torch.Tensor(bboxes), torch.Tensor(labels)
        annots = torch.cat((labels.unsqueeze(1), bboxes), dim=1)

        xy = bboxes[:, :2]
        wh = bboxes[:, 2:]

        boxes_ji = (xy * self.S).int()
        boxes_xy_cell = xy * self.S - boxes_ji
        boxes_wh_cell = wh * self.S

        boxes_xywh_cell = torch.cat((boxes_xy_cell, boxes_wh_cell), dim=1)

        annots_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))

        for (j, i), xywh, label in zip(boxes_ji.tolist(), boxes_xywh_cell, labels):
            if annots_matrix[i, j, self.C] == 0:  # first object
                annots_matrix[i, j, self.C] = 1

                # box coords
                annots_matrix[i, j, self.C + 1 : self.C + 5] = xywh
                # one hot class label
                annots_matrix[i, j, int(label)] = 1

        return image, annots_matrix
