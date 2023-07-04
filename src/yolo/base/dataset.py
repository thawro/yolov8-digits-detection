import torchvision.datasets
from typing import Literal, Optional, Callable
from src.utils.pylogger import get_pylogger
import glob
from pathlib import Path
from PIL import Image
from src.utils.utils import read_text_file
import numpy as np

log = get_pylogger(__name__)


class YOLOBaseDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        split: Literal["train", "test", "extra"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self._init_paths()
        if download:
            self.download()

        self._image_files = glob.glob(f"{self.images_path}/*.png")
        self._label_files = [
            img_file.replace("images/", "labels/").replace(".png", ".txt")
            for img_file in self._image_files
        ]

    def _init_paths(self):
        self.root = Path(self.root)
        self.labels_path = self.root / "labels" / self.split
        self.images_path = self.root / "images" / self.split

    def get_raw_data(self, idx: int):
        image_filepath = self._image_files[idx]
        labels_filepath = self._label_files[idx]

        image = np.array(Image.open(image_filepath).convert("RGB"))
        annotations = read_text_file(labels_filepath)
        return image, annotations

    def download(self):
        pass
