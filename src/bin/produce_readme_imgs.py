from src.utils.utils import DATA_PATH, ROOT
from src.data.generate_yolo_dataset import get_digit, generate_yolo_example
from src.visualization import plot_yolo_labels
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import cv2

hwd_dataset_path = DATA_PATH / "HWD+"

npy_data_path = hwd_dataset_path / "Images(500x500).npy"
npy_info_path = hwd_dataset_path / "WriterInfo.npy"
imgs_path = ROOT / "img"

data = np.load(npy_data_path)
class_ids = np.load(npy_info_path)[:, 0]

idxs = range(0, 140, 14)

imgs = data[idxs]
cut_imgs = [get_digit(img) for img in imgs]
labels = class_ids[idxs]

# raw_fig, raw_axes = plt.subplots(2, 5, figsize=(18, 7))
# cut_fig, cut_axes = plt.subplots(2, 5, figsize=(18, 7))

# raw_axes, cut_axes = raw_axes.flatten(), cut_axes.flatten()

# for i in range(len(imgs)):
#     raw_axes[i].imshow(imgs[i], cmap="gray")
#     cut_axes[i].imshow(cut_imgs[i], cmap="gray")

# raw_fig.savefig(str(imgs_path / "raw.jpeg"), bbox_inches="tight")
# cut_fig.savefig(str(imgs_path / "cut.jpeg"), bbox_inches="tight")

pre_transform = A.Compose(
    [
        A.RGBShift(r_shift_limit=128, g_shift_limit=128, b_shift_limit=128, p=0.5),
        A.ChannelShuffle(p=0.3),
    ]
)

obj_transform = A.Compose(
    [
        A.InvertImg(p=0.3),
        A.RGBShift(r_shift_limit=128, g_shift_limit=128, b_shift_limit=128, p=0.5),
        A.ChannelShuffle(p=0.3),
    ]
)

post_transforms = [
    A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=1),
]


transformed = generate_yolo_example(
    imgs=cut_imgs,
    class_ids=labels,
    nrows=3,
    ncols=3,
    p_box=0.5,
    mix_mode="and",
    imgsz=(240, 640),
    obj_transform=obj_transform,
    pre_transform=pre_transform,
    post_transforms=post_transforms,
)
transformed_img = transformed["image"]
transformed_bboxes = transformed["bboxes"]
transformed_labels = transformed["labels"]

if len(transformed_labels) > 0:
    bboxes_img = plot_yolo_labels(transformed_img, transformed_bboxes, transformed_labels)

    #  `image`, `bboxes`, `labels` keys.
    yolo_fig, yolo_axes = plt.subplots(1, 2, figsize=(18, 14))
    yolo_axes[0].imshow(transformed_img, cmap="gray")
    yolo_axes[1].imshow(bboxes_img)

    yolo_fig.savefig(str(imgs_path / "yolo_example.jpeg"), bbox_inches="tight")
else:
    print("No objects sampled")
