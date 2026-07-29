"""
yolov5s 训练 iron_steel_dataset_v2
— 自动划分 train/val (80/20)
— patience=0 关闭早退
"""
import os
import shutil
import random
from pathlib import Path

BASE    = Path(r"D:\Edge\Elcetronics competition\iron_steel_dataset_v2")
YOLOV5  = Path(r"D:\Edge\Elcetronics competition\yolov5")
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

# 2. 写入 dataset.yaml
yaml_content = f"""path: {BASE}
train: images/train
val: images/val
nc: 3
names: ['ir_sheet', 'ir_disc', 'st_ball']
"""
(BASE / "dataset.yaml").write_text(yaml_content)

# 3. 训练命令
os.chdir(YOLOV5)
train_cmd = (
    f'python train.py '
    f'--weights yolov5s.pt '
    f'--data "{BASE / "dataset.yaml"}" '
    f'--epochs 300 '
    f'--batch-size 16 '
    f'--imgsz 320 '
    f'--patience 0 '
    f'--name steel_ball_v2 '
    f'--cache'
)
print(f"\n{train_cmd}")
os.system(train_cmd)
