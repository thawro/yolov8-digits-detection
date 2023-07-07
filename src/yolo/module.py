import pytorch_lightning as pl

import torch
from torch import nn


SPLITS = ["train", "val", "test"]


class YOLODetector(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        labels: list[str],
        lr: float = 2e-5,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.labels = labels
        self.lr = lr
        self.weight_decay = weight_decay
        self.outputs = {split: [] for split in SPLITS}
        self.examples = {split: {} for split in SPLITS}
        self.logged_metrics = {}
        self.save_hyperparameters(ignore=["model", "loss_fn"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def _common_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        stage: str,
    ) -> torch.Tensor:
        images, boxes_targets = batch

        boxes_preds = self(images)  # shape: (batch, S, S, C + B * 5)
        loss = self.loss_fn(boxes_preds, boxes_targets)

        outputs = {"loss": loss, "preds": boxes_preds, "targets": boxes_targets}
        self.outputs[stage].append(outputs)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, stage="test")

    def _common_epoch_end(self, stage: str):
        outputs = self.outputs[stage]
        batches_loss = torch.tensor([output["loss"] for output in outputs])
        loss = batches_loss.mean().item()
        if self.trainer.sanity_checking:
            return loss
        loss_name = f"{stage}/loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        outputs.clear()
        self.logged_metrics.clear()

    def on_train_epoch_end(self):
        self._common_epoch_end("train")

    def on_validation_epoch_end(self):
        self._common_epoch_end("val")

    def on_test_epoch_end(self):
        self._common_epoch_end("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=7,
            threshold=0.0001,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
