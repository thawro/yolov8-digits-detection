import random
import numpy as np
from src.utils.utils import DATA_PATH
from PIL import Image
from tqdm.auto import tqdm


def main():
    random.seed(42)

    TRAIN_SIZE = 0.8
    ds_path = DATA_PATH / "HWD+"

    data = np.load(ds_path / "Images(500x500).npy").astype(np.uint8)
    info = np.load(ds_path / "WriterInfo.npy")
    class_ids = info[:, 0]

    n_all = len(data)
    all_idxs = list(range(n_all))
    n_train = int(TRAIN_SIZE * n_all)
    train_idxs = random.sample(all_idxs, k=n_train)
    val_idxs = [idx for idx in all_idxs if idx not in train_idxs]

    split_idxs = {"train": train_idxs, "val": val_idxs}

    for split, idxs in split_idxs.items():
        split_images = data[idxs]
        split_labels = class_ids[idxs]
        images_dirpath = ds_path / "images" / split
        labels_dirpath = ds_path / "labels" / split
        images_dirpath.mkdir(parents=True, exist_ok=True)
        labels_dirpath.mkdir(parents=True, exist_ok=True)

        for i, (image, label) in tqdm(
            enumerate(zip(split_images, split_labels)), total=len(split_labels)
        ):
            digit_image = Image.fromarray(image)
            digit_image.save(images_dirpath / f"{i}.png")

            txt_file = open(labels_dirpath / f"{i}.txt", "w")
            txt_file.write(str(label))
            txt_file.close()


if __name__ == "__main__":
    main()
