import os
import shutil
from pathlib import Path

from roboflow import Roboflow

try:
    from augment import augment_imgs
    from utils import load_dotenv
except ImportError:
    from lib.augment import augment_imgs
    from lib.utils import load_dotenv


def get_rb_dataset():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_SECRET"])
    project = rf.workspace("sean-mcdougall").project(
        "bp-therm-identification"
    )
    project.version(7).download(
        "yolov8-obb", location="./datasets/task1", overwrite=True
    )


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

    multiplier = 15

    augment_imgs(lcd_digits, out_dir / "lcd", multiplier=multiplier)
    augment_imgs(therm_digits, out_dir / "therm", multiplier=multiplier)


def load_datasets():
    load_dotenv()
    # get_rb_dataset()
    generate_digit_data()


if __name__ == "__main__":
    load_datasets()
