"""
yolov5s 训练 iron_steel_dataset_v2
— 自动划分 train/val (80/20)
— patience=0 关闭早退
"""
import os
import shutil
import random
from pathlib import Path

SCRIPT  = Path(__file__).resolve().parent   # 脚本所在目录（train yolov5s/）
BASE    = SCRIPT.parent / "iron_steel_dataset_v2"  # 数据集目录（与脚本目录同级仓库根下）
YOLOV5  = BASE.parent / "yolov5"                # 仓库根下相邻的 yolov5 框架（clone 后无需改路径）
IMAGES  = BASE / "images"
LABELS  = BASE / "labels"
SPLIT   = 0.8

# 1. 划分 train/val
random.seed(42)
imgs = sorted(IMAGES.glob("*.jpg"))
random.shuffle(imgs)
n_train = int(len(imgs) * SPLIT)
train_imgs = imgs[:n_train]
val_imgs   = imgs[n_train:]

print(f"Total: {len(imgs)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")

for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
    (IMAGES / subset).mkdir(exist_ok=True)
    (LABELS / subset).mkdir(exist_ok=True)
    for img in img_list:
        lbl = LABELS / (img.stem + ".txt")
        shutil.copy(img, IMAGES / subset / img.name)
        if lbl.exists():
            shutil.copy(lbl, LABELS / subset / lbl.name)

# 2. 写入 dataset.yaml（与脚本同目录）
yaml_content = f"""path: {BASE}
train: images/train
val: images/val
nc: 3
names: ['ir_sheet', 'ir_disc', 'st_ball']
"""
(SCRIPT / "dataset.yaml").write_text(yaml_content)

# 3. 训练命令
os.chdir(YOLOV5)
train_cmd = (
    f'python train.py '
    f'--weights yolov5s.pt '
    f'--data "{SCRIPT / "dataset.yaml"}" '
    f'--epochs 300 '
    f'--batch-size 16 '
    f'--imgsz 320 '
    f'--patience 0 '
    f'--name steel_ball_v2 '
    f'--cache'
)
print(f"\n{train_cmd}")
os.system(train_cmd)
