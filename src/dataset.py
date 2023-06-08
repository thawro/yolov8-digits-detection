from typing import Any, Literal, Optional, Callable
from torch import nn
from src.utils.utils import (
    DATA_PATH,
    unzip_tar_gz,
    download_file,
    save_txt_to_file,
    read_text_file,
    copy_files,
    remove_directory,
)
from src.utils.pylogger import get_pylogger
import torchvision
from PIL import Image
import torch
import glob
from pathlib import Path
import mat73
import numpy as np
from tqdm.auto import tqdm

log = get_pylogger(__name__)

SVHN_KEYS = ["label", "top", "left", "height", "width"]


def parse_annotation_svhn2yolo(
    bbox_info: dict[str, list[np.ndarray] | np.ndarray], img_filepath: str
):
    img = Image.open(img_filepath)
    w, h = img.size
    annotations = tuple(bbox_info[k] for k in SVHN_KEYS)
    if not isinstance(annotations[0], list):
        annotations = tuple([x] for x in annotations)

    annotations = [np.array(x) for x in annotations]

    txt_lines = []
    for label, top, left, height, width in zip(*annotations):
        if label == 10:  # SVHN uses 10 for digit 0, we need to use 0 instead
            label = 0

        x_center = (left + width / 2) / w
        width = width / w

        y_center = (top + height / 2) / h
        height = height / h

        # YOLOv5 format:
        # class x_center y_center width height (normalized)
        label = int(label)
        x_center, y_center, width, height = [
            x.astype(np.float32) for x in [x_center, y_center, width, height]
        ]
        txt_lines.append(" ".join([str(x) for x in [label, x_center, y_center, width, height]]))
    txt_annotation = "\n".join(txt_lines)
    return txt_annotation


class TransformWrapper(nn.Module):
    def __init__(self, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform is None:
            return x
        return self.transform(x)


class SVHNDataset(torchvision.datasets.VisionDataset):
    """
    source: http://ufldl.stanford.edu/housenumbers/
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "extra"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        transform = TransformWrapper(transform)
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.split = split
        self._init_paths()
        log.info(f"Creating {split} SVHN dataset parsed to YOLO format.")
        if download:
            self.download()

        self._parse_to_yolo()
        self._image_files = glob.glob(f"{self.images_path}/*.png")
        self._label_files = [
            img_file.replace("images/", "labels/").replace(".png", ".txt")
            for img_file in self._image_files
        ]

    def _init_paths(self):
        self.root = Path(self.root)
        self.labels_path = self.root / "labels" / self.split
        self.images_path = self.root / "images" / self.split

    def _is_yolo_parsed(self):
        if self.images_path.exists() and self.labels_path.exists():
            yolo_img_filenames = glob.glob(f"{self.images_path}/*.png")
            yolo_labels_filenames = glob.glob(f"{self.labels_path}/*.txt")

            n_yolo_img = len(yolo_img_filenames)
            if n_yolo_img == 0:
                return False
            n_yolo_labels = len(yolo_labels_filenames)

            log.info(
                f"{n_yolo_img} SVHN {self.split} images are already parsed to {n_yolo_labels} yolo labels"
            )

            if n_yolo_img == n_yolo_labels:
                return True
        return False

    def _parse_to_yolo(self):
        if self._is_yolo_parsed():
            log.info(f"SVHN {self.split} dataset is already parsed to YOLO format. Stopping.")
            return
        self.labels_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)

        log.info(f"The data was not yet parsed. Started parsing of SVHN dataset to YOLO format.")
        old_data_dir = self.root / self.split
        struct_path = str(old_data_dir / "digitStruct.mat")
        log.info(f"Reading SVHN {struct_path} annotations file (may take a while).")
        data_dict = mat73.loadmat(struct_path)
        img_filenames = data_dict["digitStruct"]["name"]
        bboxes = data_dict["digitStruct"]["bbox"]

        label_filenames = [filename.replace(".png", ".txt") for filename in img_filenames]

        copy_files(old_data_dir, self.images_path, ext=".png")
        remove_directory(old_data_dir)

        for bbox_info, img_filename, label_filename in tqdm(
            zip(bboxes, img_filenames, label_filenames),
            total=len(bboxes),
            desc="Parsing in progress",
        ):
            img_filepath = self.images_path / img_filename
            txt_annotation = parse_annotation_svhn2yolo(bbox_info, img_filepath)
            label_filepath = self.labels_path / label_filename
            save_txt_to_file(txt_annotation, label_filepath)
        log.info(f"Parsing finished.")

    def download(self):
        filename = f"{self.split}.tar.gz"
        url = f"http://ufldl.stanford.edu/housenumbers/{filename}"
        dst_path = self.root / filename
        dirpath = self.root / self.split

        if dirpath.is_dir():
            log.info(f"Using already downloaded {dirpath} dataset.")
            return

        if dst_path.is_file():
            log.info(f"Using already downloaded zip {dst_path} file.")
        else:
            download_file(url, dst_path)
        unzip_tar_gz(dst_path, self.root, remove=True)

    def get_raw_data(self, idx: int):
        image_filepath = self._image_files[idx]
        labels_filepath = self._label_files[idx]

        image = Image.open(image_filepath).convert("RGB")
        labels = read_text_file(labels_filepath)
        return image, labels

    def __getitem__(self, idx: int) -> Any:
        image, labels = self.get_raw_data(idx)
        labels = [[float(x) for x in label.split(" ")] for label in labels]
        # labels cols:
        # class x_center y_center width height
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.Tensor(labels)


if __name__ == "__main__":
    root = str(DATA_PATH / "SVHN")
    ds = SVHNDataset(root=root, split="test", download=False)
    ds = SVHNDataset(root=root, split="train", download=True)
    # ds = SVHNDataset(root=root, split="extra", download=True)
