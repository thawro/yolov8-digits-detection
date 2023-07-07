from src.yolo.v1 import YOLOv1Dataset, YOLOv1, YoloLoss
from torch.utils.data import DataLoader
from src.utils.utils import DATA_PATH, read_text_file, ROOT
from src.yolo.utils import (
    NMS,
    MAP,
    get_boxes_for_dataloader,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)

import albumentations as A
import torch.optim as optim
import torch
from tqdm.auto import tqdm


DS_NAME = "yolo_HWD+"
DS_NAME = "VOC"
DS_PATH = str(DATA_PATH / DS_NAME)
LABELS = read_text_file(f"{DS_PATH}/labels.txt")

seed = 123
torch.manual_seed(seed)

LOAD_MODEL = False

LEARNING_RATE = 2e-5
BATCH_SIZE = 16  # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
CKPT_PATH = str(ROOT / "checkpoints/ckpt.pt")
DEVICE = "cuda" if torch.cuda.is_available else "cpu"

DL_PARAMS = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

S = 7
B = 2


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


def create_dataloaders():
    transform = A.Compose(
        [A.Resize(448, 448)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
    )
    train_ds = YOLOv1Dataset(S, B, DS_PATH, "train", transform)
    val_ds = YOLOv1Dataset(S, B, DS_PATH, "val", transform)
    test_ds = YOLOv1Dataset(S, B, DS_PATH, "test", transform)

    train_dl = DataLoader(dataset=train_ds, shuffle=True, **DL_PARAMS)
    val_dl = DataLoader(dataset=val_ds, shuffle=True, **DL_PARAMS)
    test_dl = DataLoader(dataset=test_ds, shuffle=True, **DL_PARAMS)

    return train_dl, val_dl, test_dl


def main():
    train_dl, val_dl, test_dl = create_dataloaders()
    C = train_dl.dataset.C
    model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(S=S, C=C, B=B)

    if LOAD_MODEL:
        load_checkpoint(CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        if LOAD_MODEL:
            for x, y in train_dl:
                x = x.to(DEVICE)
                boxes_preds = model.inference(x)
                all_nms_boxes = model.perform_nms(boxes_preds, 0.5, 0.4)
                for i in range(8):
                    nms_boxes = all_nms_boxes[i]
                    img = x[i].permute(1, 2, 0).to("cpu")
                    plot_image(img, nms_boxes)
                exit()

        pred_boxes, target_boxes = get_boxes_for_dataloader(
            model, train_dl, iou_threshold=0.5, objectness_threshold=0.4
        )

        mean_avg_prec = MAP(C, pred_boxes, target_boxes, iou_threshold=0.5)
        print(f"Train mAP: {mean_avg_prec}")

        if mean_avg_prec > 0.5:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, path=CKPT_PATH)

        train_fn(train_dl, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
