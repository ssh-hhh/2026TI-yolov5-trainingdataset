"""从训练集选取 200 张校准图，按类别均匀覆盖"""
import shutil
import random
from pathlib import Path
from collections import defaultdict

IMG_DIR = Path(r"D:\Edge\Elcetronics competition\iron_steel_dataset_v2\images\train")
LBL_DIR = Path(r"D:\Edge\Elcetronics competition\iron_steel_dataset_v2\labels\train")
OUT_DIR = Path(r"D:\Edge\Elcetronics competition\picture")
CLASS_NAMES = ["ir_sheet", "ir_disc", "st_ball"]

random.seed(42)
OUT_DIR.mkdir(exist_ok=True)

class_images = defaultdict(list)
for lbl_path in LBL_DIR.glob("*.txt"):
    classes_in_img = set()
    for line in lbl_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        cls = int(line.split()[0])
        classes_in_img.add(cls)
    img = IMG_DIR / (lbl_path.stem + ".jpg")
    if img.exists():
        for cls in classes_in_img:
            class_images[cls].append(img)

selected = set()
target_per_class = 70

for cls in range(3):
    pool = class_images[cls]
    random.shuffle(pool)
    for p in pool:
        if p in selected:
            continue
        selected.add(p)
        if sum(1 for x in selected if x in class_images[cls]) >= target_per_class:
            break

remaining = sorted(set(class_images[0]) | set(class_images[1]) | set(class_images[2]))
random.shuffle(remaining)
for p in remaining:
    if len(selected) >= 200:
        break
    selected.add(p)

for p in selected:
    shutil.copy2(p, OUT_DIR / p.name)

print(f"选中 {len(selected)} 张 -> {OUT_DIR}")
for cls in range(3):
    cnt = sum(1 for p in selected if p in class_images[cls])
    print(f"  {CLASS_NAMES[cls]}: {cnt} 张")
