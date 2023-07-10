from src.yolo.v1 import YOLOv1, YoloLoss
from src.utils.utils import DATA_PATH, read_text_file, ROOT
from src.yolo.utils import (
    create_dataloaders,
    MAP,
    cellboxes_to_boxes,
    save_checkpoint,
    load_checkpoint,
)
from src.visualization import plot_yolo_labels
import torch.optim as optim
import torch
from tqdm.auto import tqdm
import numpy as np

DS_NAME = "yolo_HWD+"
# DS_NAME = "VOC"
DS_PATH = str(DATA_PATH / DS_NAME)
LABELS = read_text_file(f"{DS_PATH}/labels.txt")
ID2NAME = {i: label for i, label in enumerate(LABELS)}


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
IOU_THR = 0.5
OBJ_THR = 0.4


def train_loop(model, dataloader, optimizer, loss_fn):
    loop = tqdm(dataloader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    return sum(losses) / len(losses)


def val_loop(model, dataloader, loss_fn, iou_threshold, objectness_threshold, plot=False):
    all_pred_boxes = []
    all_true_boxes = []
    loss_values = []
    model.eval()
    train_idx = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        batch_size = x.shape[0]

        with torch.no_grad():
            out = model(x)

        loss = loss_fn(out, y)
        loss_values.append(loss.item())
        pred_boxes = cellboxes_to_boxes(out, S=model.S, C=model.C, B=model.B)
        pred_boxes = model.perform_nms(pred_boxes, iou_threshold, objectness_threshold)
        true_boxes = cellboxes_to_boxes(y, S=model.S, C=model.C, B=model.B)

        if plot:
            for i in range(8):
                nms_boxes = torch.tensor(pred_boxes[i]).numpy()
                img = x[i].permute(1, 2, 0).to("cpu").numpy()
                class_ids = nms_boxes[:, 0].astype(np.int64)
                obj_scores = nms_boxes[:, 1]
                boxes_xywhn = nms_boxes[:, 2:]
                plot_yolo_labels(
                    img, boxes_xywhn, class_ids, obj_scores, plot=True, id2name=ID2NAME
                )

        for idx in range(batch_size):
            for nms_box in pred_boxes[idx]:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_boxes[idx]:
                if box[1] > objectness_threshold:
                    all_true_boxes.append([train_idx] + box.tolist())

            train_idx += 1

    mean_avg_prec = MAP(model.C, all_pred_boxes, all_true_boxes, iou_threshold=iou_threshold)
    mean_loss = sum(loss_values) / len(loss_values)
    model.train()
    return {"MAP": mean_avg_prec, "loss": mean_loss}


def main():
    train_dl, val_dl, test_dl = create_dataloaders(S, B, DS_PATH, **DL_PARAMS)
    C = train_dl.dataset.C
    model = YOLOv1(S=S, B=B, C=C).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss(S=S, C=C, B=B)

    if LOAD_MODEL:
        load_checkpoint(CKPT_PATH, model, optimizer)

    for epoch in range(EPOCHS):
        val_metrics = val_loop(
            model, val_dl, loss_fn, iou_threshold=IOU_THR, objectness_threshold=OBJ_THR, plot=False
        )
        val_MAP, val_loss = val_metrics["MAP"], val_metrics["loss"]
        train_loss = train_loop(model, train_dl, optimizer, loss_fn)

        if val_MAP > 0.5:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, path=CKPT_PATH)

        print(f"Epoch {epoch}: train/loss: {train_loss:.2f},  val/loss: {val_loss:.2f}")


if __name__ == "__main__":
    main()
