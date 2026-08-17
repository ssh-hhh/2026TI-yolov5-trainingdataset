"""
生成 RDK X5 hb_mapper 校准数据 (OpenExplorer v1.2.8, march= bayes-e)。

规范 (对应官方 prepare_calibration_data 文档 + rdk_model_zoo):
  1. 每张图一个独立的 float32 二进制文件 (numpy.tofile, 无 .npy 头部)
  2. 布局: RGB, NCHW, 形状 1x3x320x320 (与 ONNX 输入一致)
  3. 值域: 0~255 原始像素值 —— 切勿归一化!
     yaml 中 norm_type='data_scale' + scale_value=1/255 会由工具链统一缩放,
     若在脚本里提前 /255 会双重归一化, 导致模型转出来但检不到目标。
  4. 数量: 20~100 张 (默认 50), 从训练集随机抽取保证代表性。
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

ROOT: Path = Path(__file__).resolve().parent.parent  # 仓库根（脚本位于 rdk_x5/，clone 后无需改路径）
SRC_DIR: Path = ROOT / "iron_steel_dataset_v2/images/train"
OUT_DIR: Path = ROOT / "rdk_x5/calibration_data_rgb_f32_320"
INPUT_SIZE: int = 320
CALIB_COUNT: int = 50
RANDOM_SEED: int = 42


def letterbox(image: np.ndarray) -> np.ndarray:
    """与 YOLOv5 推理一致的 letterbox 缩放 + 灰边填充。"""
    height, width = image.shape[:2]
    ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_w = INPUT_SIZE - resized_width
    pad_h = INPUT_SIZE - resized_height
    left = round(pad_w / 2 - 0.1)
    right = round(pad_w / 2 + 0.1)
    top = round(pad_h / 2 - 0.1)
    bottom = round(pad_h / 2 + 0.1)
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )


def main() -> None:
    images = sorted(SRC_DIR.glob("*.jpg"))
    if len(images) < CALIB_COUNT:
        raise RuntimeError(f"Need {CALIB_COUNT} images, found {len(images)} in {SRC_DIR}")

    selected = random.Random(RANDOM_SEED).sample(images, CALIB_COUNT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for img in selected:
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img}")
        rgb = cv2.cvtColor(letterbox(bgr), cv2.COLOR_BGR2RGB)
        # 保持 0~255 原始值, float32, NCHW, tofile 输出原始二进制
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None]  # (1,3,320,320)
        tensor.tofile(str(OUT_DIR / f"{img.stem}.bin"))

    print(f"校准数据生成完成: {len(selected)} 张 -> {OUT_DIR}")
    print(f"格式: float32 RGB NCHW 1x3x{INPUT_SIZE}x{INPUT_SIZE}, 值域 0~255 (未归一化)")
    print("用法: 在 hb_mapper yaml 中配置 cal_data_dir 指向此目录")


if __name__ == "__main__":
    main()
