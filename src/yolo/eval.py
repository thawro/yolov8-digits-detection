from src.yolo.v1 import YOLOv1, YoloLoss
from src.utils.utils import ROOT
from src.yolo.utils import (
    create_dataloaders,
    load_checkpoint,
)
from src.yolo.train import val_loop, DL_PARAMS, S, B, DS_PATH, IOU_THR, OBJ_THR
import torch


seed = 123
torch.manual_seed(seed)

CKPT_PATH = str(ROOT / "checkpoints/ckpt.pt")
DEVICE = "cuda" if torch.cuda.is_available else "cpu"


def main():
    train_dl, val_dl, test_dl = create_dataloaders(S, B, DS_PATH, **DL_PARAMS)
    C = train_dl.dataset.C
    model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
    loss_fn = YoloLoss(S=S, C=C, B=B)

    load_checkpoint(CKPT_PATH, model)

    dataloaders = {"train": train_dl, "val": val_dl, "test": test_dl}

    for split, dataloader in dataloaders.items():
        metrics = val_loop(
            model,
            dataloader,
            loss_fn,
            iou_threshold=IOU_THR,
            objectness_threshold=OBJ_THR,
            plot=True,
        )
        print(f"{split}/loss: {metrics['loss']:.2f}, {split}/MAP: {metrics['MAP']:.2f}")


if __name__ == "__main__":
    main()
