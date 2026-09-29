import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from roboflow import Roboflow
from utils import load_dotenv


def get_rb_dataset():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_SECRET"])
    project = rf.workspace("sean-mcdougall").project(
        "bp-therm-identification"
    )
    project.version(7).download(
        "yolov8-obb", location="./datasets/task1", overwrite=True
    )


# returns n number of augmented images for given img
# performs random rotation, scaling, and translation
def augment_img(
    img_path,
    multiplier=5,
    rot=(-15, 15),
    scale=None,
    trans=None,
):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    out_img = np.empty((multiplier, h, w, 3), dtype=np.uint8)

    for i in range(multiplier):
        # random rot, scale, trans
        angle = 0
        scaled = 1.0
        tx = 0
        ty = 0
        if rot:
            angle = np.random.uniform(rot[0], rot[1])
        if scale:
            scaled = np.random.uniform(scale[0], scale[1])
        if trans:
            tx = np.random.uniform(trans[0], trans[1])
            ty = np.random.uniform(trans[0], trans[1])

        M_rot_s = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scaled)

        M_rot_s[0, 2] += tx
        M_rot_s[1, 2] += ty

        augmented_img = cv2.warpAffine(
            img,
            M_rot_s,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_img[i] = augmented_img

    return out_img


def augment_imgs(img_dir, save_dir, multiplier=5):
    img_dir = Path(img_dir)
    save_dir = Path(save_dir)

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for img_path in img_dir.iterdir():

        folder_name = img_path.name.split(".")[0]
        out_path = save_dir / folder_name

        if not out_path.exists():
            out_path.mkdir(parents=True)

        augmented_imgs = augment_img(
            img_path, multiplier=multiplier, scale=(0.8, 1.2), trans=(-10, 10)
        )
        # copy in original img
        shutil.copy(img_path, out_path / f"{folder_name}_0.png")
        for j, aug_img in enumerate(augmented_imgs):
            cv2.imwrite(out_path / f"{folder_name}_{j+1}.png", aug_img)


# used to augment dataset for task3 so we can train da svm better
def generate_digit_data():
    base_path = Path(__file__).parent
    lcd_digits = base_path / "lcd_digits/"
    therm_digits = base_path / "therm_digits/"

    out_dir = base_path / "../datasets/task3"

    # clear current contents
    if out_dir.exists():
        shutil.rmtree(out_dir)
    else:
        out_dir.mkdir(parents=True)

    multiplier = 10

    augment_imgs(lcd_digits, out_dir / "lcd", multiplier=multiplier)
    augment_imgs(therm_digits, out_dir / "therm", multiplier=multiplier)


def load_datasets():
    load_dotenv()
    # get_rb_dataset()
    generate_digit_data()


if __name__ == "__main__":
    load_datasets()
