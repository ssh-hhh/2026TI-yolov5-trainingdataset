import os
import random
import shutil
from tqdm import tqdm

# ==================== 参数设置区域（请检查路径是否正确） ====================
# 1. 原始图片和原始标签文件的存放路径
# （如果你的原始图片在 steel_ball 文件夹外，请把这里的路径修改为实际存放路径）
SOURCE_IMAGE_DIR = r"D:\Desktop\make_dataset\iron_steel_dataset_v2\images"
SOURCE_LABEL_DIR = r"D:\Desktop\make_dataset\iron_steel_dataset_v2\labels"

# 2. 划分结果保存的根路径：保存在 steel_ball 文件夹中的新文件夹 steel_ball_dataset 下
TARGET_DATASET_DIR = r"D:\YOLO\ultralytics-8.3.163\datasets\iron_steel_datasetv2"

# 3. 划分比例设置（三者相加必须等于 1.0，即 100%）
TRAIN_RATIO = 0.7  # 训练集 70%
VAL_RATIO = 0.2    # 验证集 20%
TEST_RATIO = 0.1   # 测试集 10%

# 4. 随机种子设置（固定数值如 42，每次运行划分出的图片组合相同；若设为 None 则每次完全随机）
RANDOM_SEED = 42
# =========================================================================

def split_yolo_dataset():
    # 检查划分比例相加是否等于 1.0
    if not round(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 5) == 1.0:
        print("错误：训练集、验证集和测试集的比例相加必须等于 1.0！请检查参数设置。")
        return

    # 设置随机种子（保证实验可复现）
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 1. 在 steel_ball_dataset 下创建完整的 YOLO 目录结构
    splits = ['train', 'val', 'test']
    sub_folders = ['images', 'labels']

    # 自动生成子文件夹路径，例如 steel_ball\steel_ball_dataset\images\train
    dir_paths = {}
    for sub in sub_folders:
        for split in splits:
            path = os.path.join(TARGET_DATASET_DIR, sub, split)
            os.makedirs(path, exist_ok=True)  # 文件夹不存在时自动创建
            dir_paths[f"{sub}_{split}"] = path

    # 2. 读取图片源文件夹中的图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.exists(SOURCE_IMAGE_DIR):
        print(f"错误：图片源路径【{SOURCE_IMAGE_DIR}】不存在！请检查代码中的 SOURCE_IMAGE_DIR 设置。")
        return

    # 筛选出合法后缀的图片
    all_images = [f for f in os.listdir(SOURCE_IMAGE_DIR) if f.lower().endswith(valid_extensions)]
    total_count = len(all_images)

    if total_count == 0:
        print(f"错误：在【{SOURCE_IMAGE_DIR}】中没有找到任何图片文件！")
        return

    print(f"成功扫描到 {total_count} 张图片，准备进行随机划分...\n")

    # 3. 使用 Python 原生的 random.shuffle 进行随机打乱（不用 sklearn）
    random.shuffle(all_images)

    # 4. 按比例计算训练集、验证集、测试集的数量
    num_train = int(total_count * TRAIN_RATIO)
    num_val = int(total_count * VAL_RATIO)
    num_test = total_count - num_train - num_val  # 剩余数量归为测试集，防止舍入误差导致丢张数

    # 根据数量切分图片列表
    dataset_splits = {
        'train': all_images[:num_train],
        'val': all_images[num_train : num_train + num_val],
        'test': all_images[num_train + num_val :]
    }

    # 打印划分概况
    print("划分统计：")
    print(f"  - 训练集 (train): {len(dataset_splits['train'])} 张 ({TRAIN_RATIO*100:.0f}%)")
    print(f"  - 验证集 (val):   {len(dataset_splits['val'])} 张 ({VAL_RATIO*100:.0f}%)")
    print(f"  - 测试集 (test):  {len(dataset_splits['test'])} 张 ({TEST_RATIO*100:.0f}%)\n")

    # 5. 开始复制图片和对应的 txt 标签
    for split_name, img_list in dataset_splits.items():
        missing_label_count = 0  # 统计未找到标签的图片数量

        # tqdm 实时进度条
        pbar = tqdm(img_list, desc=f"正在复制 [{split_name}] 集", unit="组")
        for img_name in pbar:
            # (1) 复制图片到目标 images/train (或 val, test)
            src_img_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
            dst_img_path = os.path.join(dir_paths[f"images_{split_name}"], img_name)
            shutil.copy2(src_img_path, dst_img_path)  # copy2: 仅复制，不影响原文件

            # (2) 匹配并复制对应的 txt 标签文件到 labels/train (或 val, test)
            base_name = os.path.splitext(img_name)[0]  # 获得文件名（如 abc）
            label_name = f"{base_name}.txt"
            src_label_path = os.path.join(SOURCE_LABEL_DIR, label_name)

            # 判断标签 txt 是否存在
            if os.path.exists(src_label_path):
                dst_label_path = os.path.join(dir_paths[f"labels_{split_name}"], label_name)
                shutil.copy2(src_label_path, dst_label_path)
            else:
                missing_label_count += 1

        if missing_label_count > 0:
            print(f"  [提示] [{split_name}] 集中有 {missing_label_count} 张图片未找到对应的 .txt 标签。")

    print(f"\n全部划分并复制完成！新的数据集已成功生成在：")
    print(TARGET_DATASET_DIR)

if __name__ == "__main__":
    split_yolo_dataset()