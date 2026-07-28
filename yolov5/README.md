# 电赛钢珠检测 — YOLOv5n

基于 YOLOv5 v7.0，使用 yolov5n 模型训练钢珠（steel-ball）目标检测。

## 目录结构

```
电赛/
├── 电赛钢珠数据集/          # 数据集（YOLOv5 格式）
│   ├── data.yaml            # 数据集配置
│   ├── train/               # 351 张训练图
│   ├── valid/               # 30 张验证图
│   └── test/                # 16 张测试图
│
└── yolov5/                  # 训练代码
    ├── train.py             # 训练脚本
    ├── val.py               # 验证脚本
    ├── detect.py            # 推理脚本
    ├── export.py            # 模型导出脚本
    ├── yolov5n.pt           # 预训练权重
    ├── models/              # 模型定义
    ├── utils/               # 工具函数
    └── data/hyps/           # 超参数配置
```

## 环境配置

```bash
cd D:\Edge\电赛\yolov5
pip install -r requirements.txt
```

核心依赖：Python >= 3.7，PyTorch >= 1.7

## 数据集

- **类别**：1 类 — `steel-ball`
- **图片尺寸**：512×512（Roboflow 预处理）
- **标注格式**：YOLOv5（归一化 xywh）
- **来源**：Roboflow 导出，v1 2026-07-25

## 流水线

### 1. 训练

```bash
python train.py \
    --weights yolov5n.pt \
    --data "D:/Edge/电赛/电赛钢珠数据集/data.yaml" \
    --epochs 300 \
    --batch-size 16 \
    --imgsz 512 \
    --hyp data/hyps/hyp.scratch-low.yaml \
    --patience 50 \
    --cache \
    --name steel-ball
```

| 参数 | 说明 |
|------|------|
| `--weights yolov5n.pt` | 基于 COCO 预训练的 yolov5n（1.9M 参数） |
| `--imgsz 512` | 与数据集预处理尺寸一致 |
| `--hyp hyp.scratch-low.yaml` | nano 模型推荐超参，轻量数据增强 |
| `--patience 50` | 50 epoch 无提升则早停 |
| `--cache` | 缓存图像到内存，加速训练 |

**输出**：`runs/train/steel-ball/weights/best.pt`

### 2. 验证

```bash
python val.py \
    --weights runs/train/steel-ball/weights/best.pt \
    --data "D:/Edge/电赛/电赛钢珠数据集/data.yaml" \
    --imgsz 512 \
    --task test
```

输出 P（精确率）、R（召回率）、mAP@0.5、mAP@0.5:0.95。

### 3. 推理

```bash
# 单张图片
python detect.py \
    --weights runs/train/steel-ball/weights/best.pt \
    --source your_image.jpg \
    --data "D:/Edge/电赛/电赛钢珠数据集/data.yaml" \
    --imgsz 512 \
    --save-txt --save-conf

# 测试集批量推理
python detect.py \
    --weights runs/train/steel-ball/weights/best.pt \
    --source "D:/Edge/电赛/电赛钢珠数据集/test/images" \
    --data "D:/Edge/电赛/电赛钢珠数据集/data.yaml" \
    --imgsz 512
```

**输出**：`runs/detect/exp/`（标注图 + 可选 txt 标签）

### 4. 导出

```bash
python export.py \
    --weights runs/train/steel-ball/weights/best.pt \
    --data "D:/Edge/电赛/电赛钢珠数据集/data.yaml" \
    --imgsz 512 512 \
    --include onnx torchscript
```

导出 ONNX 和 TorchScript 格式用于部署。生成文件与 `best.pt` 同目录。

## 常用调参

| 场景 | 调整 |
|------|------|
| 显存不足 | `--batch-size 8` 或 `--batch-size 4` |
| 训练太慢 | `--epochs 100` |
| 小目标难检测 | `--imgsz 640` |
| 需要更强模型 | 换 `yolov5s.pt`，`--hyp hyp.scratch-high.yaml` |
| 断点续训 | `--resume` |
