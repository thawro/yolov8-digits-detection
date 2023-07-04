import numpy as np
import random
import cv2
import albumentations as A
from albumentations.core.composition import TransformsSeqType
from src.utils.utils import DATA_PATH, read_text_file, save_txt_to_file
from src.utils.pylogger import get_pylogger
from pathlib import Path
from typing import Literal
from PIL import Image
from tqdm.auto import tqdm
import glob

log = get_pylogger(__name__)

# digits dataset from: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9702948/
# https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or


def get_digit(image: np.ndarray):
    """Cut digit from image (remove white background)"""
    cols = np.where(~np.all(image == 255, axis=0))[0]
    rows = np.where(~np.all(image == 255, axis=1))[0]
    return image[rows][:, cols]


def save_single_objects(ds_path: Path):
    """Iterate over digits images localized in dataset path, remove background
    and save each digit and label in separate files.
    Args:
        ds_path (Path): Path to the objects dataset
    """
    log.info(f"Loading {str(ds_path)} dataset")
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
    for i, (image, label) in tqdm(enumerate(zip(data, class_ids)), total=len(class_ids)):
        digit = get_digit(image)
        digit_image = Image.fromarray(digit)
        digit_image.save(images_dirpath / f"{i}.png")

        txt_file = open(labels_dirpath / f"{i}.txt", "w")
        txt_file.write(str(label))
        txt_file.close()


def generate_yolo_example(
    imgs: list[np.ndarray],
    class_ids: list[str],
    nrows: int = 3,
    ncols: int = 3,
    p_box: float = 0.5,
    mix_mode: Literal["and", "or", "equal", "random"] = "random",
    imgsz: tuple[int, int] = (640, 640),
    pre_transform: A.Compose | None = None,
    obj_transform: A.Compose | None = None,
    post_transforms: TransformsSeqType | None = None,
) -> dict[str, np.ndarray]:
    """Generate image for further yolo training.
    First a grid `nrows x ncols` is created, then for each `i` cell in the grid
    an `imgs[i]` image is sampled with `p_box` probability and randomly
    placed in that cell.
    This function is made for gray digits dataset (uses white background)
    and applies `&` operator between a digit box and a background

    Args:
        imgs (list[np.ndarray]): List of single objects images
        class_ids (list[str]): List of class ids for each object
        nrows (int, optional): Number of grid rows. Defaults to 3.
        ncols (int, optional): Number of grid cols. Defaults to 3.
        p_box (float, optional): Probability that an object will be sampled
            at [row, col] position. Defaults to 0.5.
        mix_mode (Literal["and", "or", "equal", "random"]): How to mix object box with background image.
            "and" applies `&` operator, "or" applier `|` operator, "equal" sets object box directly,
            "random" randomly choses one of ["and", "or", "equal"] for each box. Default to "random".
        imgsz (tuple[int, int], optional): Desired image size ([height, width]).
            Defaults to (640, 640).
        pre_transform (A.Compose, optional): transform applied to the background of the grid
            before sampling single objects.
        obj_transform (A.Compose, optional): transform applied to each single object put on the grid.
        post_transforms (TransformsSeqType, optional): transforms applied to the grid
            after objects sampling.

    Returns:
        dict[str, np.ndarray]: Example YOLO training imput, that is a `dict` with
            `image`, `bboxes`, `labels` keys.
    """

    def mix_box_with_bg(mode: Literal["and", "or", "equal", "random"], bg_patch, box_img):
        if mode == "and":
            return bg_patch & box_img
        elif mode == "or":
            return bg_patch | box_img
        elif mode == "equal":
            return box_img
        elif mode == "random":
            return mix_box_with_bg(random.choice(["and", "or", "equal"]), bg_patch, box_img)

    H, W = imgsz

    box_h, box_w = H // nrows, W // ncols

    margin_h = box_h // 2
    margin_w = box_w // 2

    BG_H, BG_W = H + margin_h * 2, W + margin_w * 2
    bg_img = np.ones((BG_H, BG_W, 3)).astype(np.uint8) * 255
    if pre_transform is not None:
        bg_img = pre_transform(image=bg_img)["image"]

    bboxes = []
    labels = []
    # for i, (img, class_id) in enumerate(zip(imgs, class_ids)):
    for i in range(nrows * ncols):  # make sure that only nrows * ncols images are used
        if random.random() > p_box:
            continue

        img, class_id = imgs[i], class_ids[i]
        row, col = i // ncols, i % ncols
        box_x_min = col * box_w + margin_w
        box_x_max = box_x_min + box_w
        box_y_min = row * box_h + margin_h
        box_y_max = box_y_min + box_h

        h, w, *_ = img.shape
        w_ratio = box_w / w
        h_ratio = box_h / h

        if h_ratio < w_ratio:
            fy = box_h / h
            fx = fy
        else:
            fx = box_w / w
            fy = fx
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        h, w, *c = img.shape
        if len(c) == 0:  # gray -> RGB
            # digits dataset is in gray, so repeating RGB to get 3 channels for YOLO
            img = img[..., np.newaxis]
            img = np.repeat(img, 3, axis=2)

        x_center = random.randint(box_x_min + box_w // 3, box_x_max - box_w // 3)
        y_center = random.randint(box_y_min + box_h // 3, box_y_max - box_h // 3)

        left = w // 2
        right = w - left

        bottom = h // 2
        top = h - bottom

        x_min, x_max = x_center - left, x_center + right
        y_min, y_max = y_center - bottom, y_center + top
        if obj_transform is not None:
            img = obj_transform(image=img)["image"]
        bg_patch = bg_img[y_min:y_max, x_min:x_max]
        bg_img[y_min:y_max, x_min:x_max] = mix_box_with_bg(mix_mode, bg_patch, img)

        x_center_n = x_center / BG_W
        y_center_n = y_center / BG_H
        w_n = w / BG_W
        h_n = h / BG_H

        bboxes.append([x_center_n, y_center_n, w_n, h_n])
        labels.append(class_id)

    bboxes = np.array(bboxes)
    labels = np.array(labels)
    transforms = [A.augmentations.crops.transforms.CenterCrop(H, W)]
    if post_transforms is not None:
        transforms.extend(post_transforms)
    transform = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["labels"]),
    )
    transformed = transform(image=bg_img, bboxes=bboxes, labels=labels)
    return transformed


def generate_yolo_split_data(
    images_filepaths: np.ndarray,
    new_ds_path: Path,
    split: str,
    n_images: int = 1,
    nrows_low: int = 2,
    nrows_high: int = 8,
    ncols_low: int = 2,
    ncols_high: int = 8,
    p_box: float = 0.5,
    mix_mode: Literal["and", "or", "equal", "random"] = "random",
    imgsz: tuple[int, int] = (640, 640),
    pre_transform: A.Compose | None = None,
    obj_transform: A.Compose | None = None,
    post_transforms: TransformsSeqType | None = None,
):
    """Generates yolo-like dataset using images from `images_filepaths`
    and save this dataset to `new_ds_path`.
    Objects labels must be placed in labels/ directory and stored in .txt files.

    Grid size (nrows x ncols) is sampled for each example using randint(low, high).

    Args:
        images_filepaths (np.ndarray): Filepaths of single object images.
        new_ds_path (Path): Directory path for the new dataset.
        split (str): Split name.
        n_images (int, optional): How much examples to generate. Defaults to 1.
        nrows_low (int, optional): Low boundary of nrows for grid sampling. Defaults to 2.
        nrows_high (int, optional): High boundary of nrows for grid sampling. Defaults to 8.
        ncols_low (int, optional): Low boundary of ncols for grid sampling. Defaults to 2.
        ncols_high (int, optional): High boundary of ncols for grid sampling. Defaults to 8.
        p_box (float, optional): Probability that an object will be sampled
            at [row, col] position. Defaults to 0.5.
        mix_mode (Literal["and", "or", "equal", "random"]): How to mix object box with background image.
            "and" applies `&` operator, "or" applier `|` operator, "equal" sets object box directly,
            "random" randomly choses one of ["and", "or", "equal"] for each box. Default to "random".
        imgsz (tuple[int, int], optional): Desired image size ([height, width]).
            Defaults to (640, 640).
        pre_transform (A.Compose, optional): transform applied to the background of the grid
            before sampling single objects.
        obj_transform (A.Compose, optional): transform applied to each single object put on the grid.
        post_transforms (TransformsSeqType, optional): transforms applied to the grid
            after objects sampling.
    """
    labels_filepaths = np.array(
        [path.replace("images/", "labels/").replace(".png", ".txt") for path in images_filepaths]
    )

    idxs = list(range(len(images_filepaths)))

    dst_images_dirpath = new_ds_path / "images" / split
    dst_labels_dirpath = new_ds_path / "labels" / split

    dst_images_dirpath.mkdir(exist_ok=True, parents=True)
    dst_labels_dirpath.mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(n_images), desc=f"Generating YOLO labeled images for {split} split"):
        nrows = random.randint(nrows_low, nrows_high)
        ncols = random.randint(ncols_low, ncols_high)
        n_examples = nrows * ncols
        random_idxs = random.choices(idxs, k=n_examples)
        imgs = [np.asarray(Image.open(filepath)) for filepath in images_filepaths[random_idxs]]
        labels = [read_text_file(filepath)[0] for filepath in labels_filepaths[random_idxs]]
        transformed = generate_yolo_example(
            imgs,
            labels,
            nrows=nrows,
            ncols=ncols,
            p_box=p_box,
            mix_mode=mix_mode,
            imgsz=imgsz,
            pre_transform=pre_transform,
            obj_transform=obj_transform,
            post_transforms=post_transforms,
        )
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        classes = transformed["labels"]
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


def generate_yolo_dataset(
    old_ds_path: Path,
    new_ds_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    n_images: int = 1000,
):
    """Generates yolo dataset for all splits.

    Single objects images must be placed in `old_ds_path/images` and stored in `.png` files.
    Objects labels must be placed in `old_ds_path/labels` directory and stored in `.txt` files.

    Args:
        old_ds_path (Path): Directory path of dataset with source images and labels.
        new_ds_path (Path): Directory path of new dataset.
        train_ratio (float, optional): Ratio of source images used to
            create train dataset. Defaults to 0.8.
        val_ratio (float, optional): Ratio of source images used to
            create val dataset. Defaults to 0.1.
        n_images (int, optional): Total number of images to generate.
            Split ratios are applied for that aswell

    """
    test_ratio = 1 - train_ratio - val_ratio
    all_images_filepaths = np.array(glob.glob(str(old_ds_path / "images/*")))
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

    train_n_images = int(train_ratio * n_images)
    val_n_images = int(val_ratio * n_images)
    test_n_images = int(test_ratio * n_images)

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

    kwargs = dict(
        new_ds_path=new_ds_path,
        nrows_low=1,
        nrows_high=8,
        ncols_low=1,
        ncols_high=8,
        p_box=0.5,
        mix_mode="and",
        imgsz=(640, 640),
        pre_transform=pre_transform,
        obj_transform=obj_transform,
        post_transforms=post_transforms,
    )

    generate_yolo_split_data(
        train_image_filepaths, split="train", n_images=train_n_images, **kwargs
    )
    generate_yolo_split_data(val_image_filepaths, split="val", n_images=val_n_images, **kwargs)
    generate_yolo_split_data(test_image_filepaths, split="test", n_images=test_n_images, **kwargs)


if __name__ == "__main__":
    old_ds_path = DATA_PATH / "HWD+"
    new_ds_path = DATA_PATH / "yolo_HWD+"

    # save_single_objects(old_ds_path)

    generate_yolo_dataset(
        old_ds_path,
        new_ds_path,
        n_images=10_000,
    )
