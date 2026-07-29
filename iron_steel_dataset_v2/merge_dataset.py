import os
import re
import shutil

# ==================== 1. 路径配置 ====================
# 数据集 1 路径（基础数据集，文件名 100% 保持原样）
DATASET1_DIR = r"D:\Desktop\make_dataset\iron_steel_dataset"

# 数据集 2 路径（需要重命名接在后面的 steel_ball 数据集）
DATASET2_DIR = r"D:\Desktop\make_dataset\steel_ball"

# 合并后生成的新数据集路径
OUTPUT_DIR = r"D:\Desktop\make_dataset\iron_steel_dataset_v2"

# ==================== 2. 类别 ID 自动转换配置 ====================
# ⚠️ 关键设置：把 steel_ball 标签 txt 里的单类 ID '0' 自动修改为 3 类别里的 '2' (st_ball)
STEEL_BALL_CLASS_REMAP = {0: 2}


def parse_file_number(filename):
    """
    辅助函数：解析文件名，提取前缀（如 '001_'）和后面的数字（如 192）
    示例：'001_00192.jpg' -> 返回 前缀 '001_', 数字 192, 数字位数 5
    """
    name_without_ext = os.path.splitext(filename)[0]
    # 正则表达式匹配：(前缀_)(数字)
    match = re.match(r"^(.*_)?(\d+)$", name_without_ext)
    if match:
        prefix = match.group(1) or "001_"
        num_str = match.group(2)
        return prefix, int(num_str), len(num_str)
    return "001_", 0, 5


def merge_datasets():
    # 构造新数据集的输出路径
    out_images_dir = os.path.join(OUTPUT_DIR, "images")
    out_labels_dir = os.path.join(OUTPUT_DIR, "labels")

    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    # ----------------------------------------------------
    # 第一步：复制数据集 1 (iron_steel_dataset)
    # 规则：保持原文件名 1:1 复制，不修改任何文件名！
    # ----------------------------------------------------
    ds1_img_dir = os.path.join(DATASET1_DIR, "images")
    ds1_lbl_dir = os.path.join(DATASET1_DIR, "labels")

    if not os.path.exists(ds1_img_dir):
        print(f"❌ 错误：未找到数据集 1 的图片目录 {ds1_img_dir}")
        return

    ds1_files = [f for f in os.listdir(ds1_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    ds1_files.sort()

    print(f"📦 [1/2] 正在复制数据集 1 ({DATASET1_DIR})...")
    print(f"     保持原文件名 1:1 原样复制（共 {len(ds1_files)} 张）...")

    max_num = 0
    default_prefix = "001_"
    digits_len = 5

    for img_name in ds1_files:
        # 解析文件名中的序号，记录最大序号
        prefix, num, d_len = parse_file_number(img_name)
        if num > max_num:
            max_num = num
            default_prefix = prefix
            digits_len = d_len

        # 旧文件与新文件路径
        old_img_path = os.path.join(ds1_img_dir, img_name)
        old_lbl_name = os.path.splitext(img_name)[0] + ".txt"
        old_lbl_path = os.path.join(ds1_lbl_dir, old_lbl_name)

        new_img_path = os.path.join(out_images_dir, img_name)
        new_lbl_path = os.path.join(out_labels_dir, old_lbl_name)

        # 1:1 直接复制，不修改文件名
        shutil.copy2(old_img_path, new_img_path)

        if os.path.exists(old_lbl_path):
            shutil.copy2(old_lbl_path, new_lbl_path)
        else:
            open(new_lbl_path, 'w').close()

    print(f"✅ 数据集 1 复制完成！")
    print(f"   数据集 1 的最大序号为: {max_num}，前缀格式为: '{default_prefix}'")

    # ----------------------------------------------------
    # 第二步：处理数据集 2 (steel_ball)
    # 规则：保持前缀，接在 max_num 后面顺序编号 + 修正 txt 里的类别 ID(0->2)
    # ----------------------------------------------------
    ds2_img_dir = os.path.join(DATASET2_DIR, "images")
    ds2_lbl_dir = os.path.join(DATASET2_DIR, "labels")

    if not os.path.exists(ds2_img_dir):
        print(f"❌ 错误：未找到数据集 2 的图片目录 {ds2_img_dir}")
        return

    ds2_files = [f for f in os.listdir(ds2_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    ds2_files.sort()

    # 从数据集 1 的最大序号加 1 开始编号
    current_num = max_num + 1

    print(f"\n📦 [2/2] 正在处理数据集 2 ({DATASET2_DIR})...")
    print(f"     接在序号 {current_num} 之后顺延重命名（共 {len(ds2_files)} 张）...")

    for img_name in ds2_files:
        ext = os.path.splitext(img_name)[1]

        # 构造拼接后的新文件名，例如 001_00193.jpg
        new_base_name = f"{default_prefix}{current_num:0{digits_len}d}"
        new_img_name = new_base_name + ext
        new_lbl_name = new_base_name + ".txt"

        old_img_path = os.path.join(ds2_img_dir, img_name)
        old_lbl_name = os.path.splitext(img_name)[0] + ".txt"
        old_lbl_path = os.path.join(ds2_lbl_dir, old_lbl_name)

        new_img_path = os.path.join(out_images_dir, new_img_name)
        new_lbl_path = os.path.join(out_labels_dir, new_lbl_name)

        # 1. 复制图片并重命名
        shutil.copy2(old_img_path, new_img_path)

        # 2. 处理标签文件：复制重命名 + 将类别编号 0 改为 2
        if os.path.exists(old_lbl_path):
            with open(old_lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls_id = int(parts[0])
                # 把类别 ID 0 自动修改为 2
                new_cls_id = STEEL_BALL_CLASS_REMAP.get(old_cls_id, old_cls_id)
                parts[0] = str(new_cls_id)
                new_lines.append(" ".join(parts) + "\n")

            # 写入新的 txt 文件
            with open(new_lbl_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        else:
            open(new_lbl_path, 'w').close()

        current_num += 1

    print("\n==================================================")
    print("🎉 两个数据集合并成功！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📊 数据集 1 文件范围: {ds1_files[0]} ~ {ds1_files[-1]}")
    print(
        f"📊 数据集 2 接续编号: {default_prefix}{max_num + 1:0{digits_len}d} ~ {default_prefix}{current_num - 1:0{digits_len}d}")
    print(f"🎯 标签转换提醒: steel_ball 数据集里 txt 的类别 ID 已自动从 0 改为了 2 (st_ball)")
    print("==================================================")


if __name__ == "__main__":
    merge_datasets()