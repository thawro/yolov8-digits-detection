import numpy as np
import random
import cv2
import albumentations as A
from src.utils.utils import DATA_PATH, read_text_file, save_txt_to_file
from PIL import Image
from pathlib import Path
from src.utils.pylogger import get_pylogger
from tqdm.auto import tqdm
import glob

log = get_pylogger(__name__)

# digits dataset from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9702948/
# https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or


def get_digit(img):
    cols = np.where(~np.all(img == 255, axis=0))[0]
    rows = np.where(~np.all(img == 255, axis=1))[0]
    return img[rows][:, cols]


def save_single_digits():
    ds_path = DATA_PATH / "HWD+"
    log.info("Loading HWD+ dataset")
    data = np.load(ds_path / "Images(500x500).npy").astype(np.uint8)
    info = np.load(ds_path / "WriterInfo.npy")

    # get digit ids only (class labels)
    class_ids = info[:, 0]

    # binary threshold
    data[data > 128] = 255
    data[data <= 128] = 0

    images_dirpath = ds_path / "images"
    labels_dirpath = ds_path / "labels"

    images_dirpath.mkdir(exist_ok=True, parents=True)
    labels_dirpath.mkdir(exist_ok=True, parents=True)

    log.info(f"Extracting digits from HWD+ images and saving to files in {str(ds_path)} directory")
    for i, (img, label) in tqdm(enumerate(zip(data, class_ids)), total=len(class_ids)):
        digit = get_digit(img)
        digit_image = Image.fromarray(digit)
        digit_image.save(images_dirpath / f"{i}.png")

        txt_file = open(labels_dirpath / f"{i}.txt", "w")
        txt_file.write(str(label))
        txt_file.close()


def generate_yolo_example(imgs, class_ids, nrows=3, ncols=3, p_box=0.5, imgsz=(640, 640)):
    H, W = imgsz

    box_h, box_w = H // nrows, W // ncols

    margin_h = box_h // 2
    margin_w = box_w // 2

    BG_H, BG_W = H + margin_h * 2, W + margin_w * 2
    bg_img = np.ones((BG_H, BG_W)).astype(np.uint8) * 255

    bboxes = []
    labels = []
    for i, (img, class_id) in enumerate(zip(imgs, class_ids)):
        row, col = i // ncols, i % ncols
        if random.random() > p_box:
            continue
        box_x_min = col * box_w + margin_w
        box_x_max = box_x_min + box_w
        box_y_min = row * box_h + margin_h
        box_y_max = box_y_min + box_h

        digit = get_digit(img)
        h, w = digit.shape
        w_ratio = box_w / w
        h_ratio = box_h / h

        if h_ratio < w_ratio:
            fy = box_h / h
            fx = fy
        else:
            fx = box_w / w
            fy = fx
        digit = cv2.resize(digit, (0, 0), fx=fx, fy=fy)
        h, w = digit.shape

        x_center = random.randint(box_x_min + box_w // 3, box_x_max - box_w // 3)
        y_center = random.randint(box_y_min + box_h // 3, box_y_max - box_h // 3)

        left = w // 2
        right = w - left

        bottom = h // 2
        top = h - bottom

        x_min, x_max = x_center - left, x_center + right
        y_min, y_max = y_center - bottom, y_center + top
        patch = bg_img[y_min:y_max, x_min:x_max]
        bg_img[y_min:y_max, x_min:x_max] = patch & digit
        x_center_n = x_center / BG_W
        y_center_n = y_center / BG_H
        w_n = w / BG_W
        h_n = h / BG_H

        bboxes.append([x_center_n, y_center_n, w_n, h_n])

        labels.append(class_id)

    bboxes = np.array(bboxes)
    labels = np.array(labels)

    transform = A.Compose(
        [A.augmentations.crops.transforms.CenterCrop(H, W)],
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
    )
    transformed = transform(image=bg_img, bboxes=bboxes, labels=labels)
    image = transformed["image"]
    bboxes = transformed["bboxes"]
    classes = transformed["labels"]
    return image, bboxes, classes


def generate_yolo_split_data(images_filepaths, split, n_images=1):
    # images_filepaths = np.array(glob.glob(str(DATA_PATH / "HWD+/images/*")))
    labels_filepaths = np.array(
        [path.replace("images/", "labels/").replace(".png", ".txt") for path in images_filepaths]
    )

    idxs = list(range(len(images_filepaths)))

    ds_path = DATA_PATH / "yolo_HWD+"
    dst_images_dirpath = ds_path / "images" / split
    dst_labels_dirpath = ds_path / "labels" / split

    dst_images_dirpath.mkdir(exist_ok=True, parents=True)
    dst_labels_dirpath.mkdir(exist_ok=True, parents=True)

    p_box = 0.5
    imgsz = (640, 640)

    for i in tqdm(range(n_images), desc=f"Generating YOLO labeled images for {split} split"):
        nrows = random.randint(2, 8)
        ncols = random.randint(2, 8)
        n_examples = nrows * ncols
        random_idxs = random.choices(idxs, k=n_examples)
        imgs = [np.asarray(Image.open(filepath)) for filepath in images_filepaths[random_idxs]]
        labels = [read_text_file(filepath)[0] for filepath in labels_filepaths[random_idxs]]
        image, bboxes, classes = generate_yolo_example(
            imgs, labels, nrows=nrows, ncols=ncols, p_box=p_box, imgsz=imgsz
        )
        image = image[..., np.newaxis]
        image = np.repeat(image, 3, axis=2)
        if len(bboxes) == 0:
            continue
        Image.fromarray(image).save(str(dst_images_dirpath / f"{i}.png"))
        txt_lines = []
        for bbox, class_id in zip(bboxes, classes):
            x_center, y_center, width, height = bbox
            txt_lines.append(
                " ".join([str(x) for x in [class_id, x_center, y_center, width, height]])
            )
        txt_annotation = "\n".join(txt_lines)
        save_txt_to_file(txt_annotation, dst_labels_dirpath / f"{i}.txt")


def generate_yolo_dataset(train_ratio=0.8, val_ratio=0.1):
    all_images_filepaths = np.array(glob.glob(str(DATA_PATH / "HWD+/images/*")))
    N = len(all_images_filepaths)
    all_idxs = list(range(N))
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train_idxs = random.choices(all_idxs, k=n_train)
    all_idxs = list(set(all_idxs).difference(set(train_idxs)))

    val_idxs = random.choices(all_idxs, k=n_val)
    test_idxs = list(set(all_idxs).difference(set(val_idxs)))

    train_image_filepaths = all_images_filepaths[train_idxs]
    val_image_filepaths = all_images_filepaths[val_idxs]
    test_image_filepaths = all_images_filepaths[test_idxs]

    generate_yolo_split_data(train_image_filepaths, "train", n_images=500)
    generate_yolo_split_data(val_image_filepaths, "val", n_images=50)
    generate_yolo_split_data(test_image_filepaths, "test", n_images=50)


if __name__ == "__main__":
    # save_single_digits()
    generate_yolo_dataset()
