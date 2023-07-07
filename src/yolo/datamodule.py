import pytorch_lightning as pl


import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.yolo.base.dataset import YOLOBaseDataset


class YOLODataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_ds: YOLOBaseDataset,
        val_ds: YOLOBaseDataset,
        test_ds: YOLOBaseDataset,
        batch_size: int = 64,
        num_workers: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

    def state_dict(self):
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict):
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
