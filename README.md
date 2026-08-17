# 2026 电子设计竞赛 —— 钢材目标检测 (YOLOv5s)

基于 **YOLOv5s** 的钢材表面目标检测项目，检测三类目标：**铁片 (`ir_sheet`)**、**铁盘 (`ir_disc`)**、**钢珠 (`st_ball`)**。
训练→验证→ONNX 导出→INT8 量化全流程闭环，目标部署平台为 **地平线 RDK X5** 边缘计算板。

---

## 目录

- [特性](#特性)
- [目录结构](#目录结构)
- [处理流程](#处理流程)
- [文件说明](#文件说明)
- [运行环境配置](#运行环境配置)
- [快速开始](#快速开始)
- [训练结果](#训练结果)
- [部署到 RDK X5](#部署到-rdk-x5)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 特性

- ✅ 基于官方 [ultralytics/yolov5](https://github.com/ultralytics/yolov5) 框架，改动最小
- ✅ 3 类目标检测：铁片 / 铁盘 / 钢珠（`nc=3`）
- ✅ 数据集自动划分 train/val（80% / 20%），一键脚本
- ✅ 训练配置可复现：`yolov5s` + 320px + 300 epochs，mAP@0.5 达 **0.990**
- ✅ ONNX 导出 + 静态 INT8 量化（QDQ），模型体积 26.9MB → **7.1MB**
- ✅ FP32 / INT8 双版本在验证集上的精度、延迟对比
- 🎯 面向 RDK X5 部署（.bin 转换，见[部署](#部署到-rdk-x5)）

---

## 目录结构

```
Elcetronics competition/
├── iron_steel_dataset_v2/          # 数据集 + 训练脚本
│   ├── images/
│   │   ├── train/                  # 训练集图片 (542 张)
│   │   ├── val/                    # 验证集图片 (136 张)
│   │   └── *.jpg                   # 原始图片 (678 张，划分前的源文件)
│   ├── labels/
│   │   ├── train/                  # 训练集标注 (542 个 .txt)
│   │   ├── val/                    # 验证集标注 (136 个 .txt)
│   │   ├── *.txt                   # 原始标注 (678 个，与原始图片同名)
│   │   ├── classes.txt             # 类别清单 (ir_sheet / ir_disc / st_ball)
│   │   ├── train.cache             # 训练标签缓存（可自动重建）
│   │   ├── val.cache               # 验证标签缓存（可自动重建）
│   │   └── split.py                # ⚠️ 遗留的旧划分脚本（路径已失效，勿用）
│   ├── dataset.yaml                # YOLOv5 数据集配置（路径/类别数/类别名）
│   └── train_yolov5s.py            # 一键脚本：划分数据集 + 启动训练
│
├── yolov5/                         # YOLOv5 框架（官方仓库 + 自定义训练参数）
│   ├── data/hyps/                  # 超参数配置（学习率、增强策略等）
│   ├── models/                     # 网络结构定义
│   ├── utils/                      # 工具库（数据加载、指标、绘图等）
│   ├── runs/
│   │   ├── train/steel_ball_v1/    # 训练产物
│   │   │   ├── weights/
│   │   │   │   ├── best.pt         # 最优权重 (13.6MB)
│   │   │   │   ├── last.pt         # 最后一轮权重 (13.6MB)
│   │   │   │   ├── best.onnx       # 导出 ONNX (26.9MB)
│   │   │   │   └── best_int8.onnx  # INT8 量化模型 (7.1MB) ← 部署用
│   │   │   ├── results.csv         # 逐 epoch 训练/验证指标
│   │   │   └── *.png               # 曲线图（PR、P、R、混淆矩阵等）
│   │   ├── val/fp32_onnx/          # FP32 ONNX 在验证集上的结果
│   │   └── val/int8_onnx/          # INT8 ONNX 在验证集上的结果
│   ├── train.py                    # 训练主程序
│   ├── val.py                      # 验证主程序
│   ├── detect.py                   # 推理主程序
│   ├── export.py                   # 模型导出（.pt → .onnx 等）
│   ├── yolov5s.pt                  # COCO 预训练权重（训练起点）
│   └── requirements.txt            # Python 依赖清单
│
├── quantize_int8.py                # ONNX Runtime 静态 INT8 量化脚本
├── select_calib.py                 # ⚠️ 遗留：校准图选取脚本（已被量化脚本内置逻辑取代）
├── picture/                        # ⚠️ 遗留：旧校准图副本（200 张，不入库）
├── rdk_x5/                         # RDK X5 部署转换工具链 (.onnx → .bin)
│   ├── export_rdk_onnx.py          # 按地平线规范导出 ONNX（NHWC 3 特征头）
│   ├── prepare_calib_data.py       # 生成 hb_mapper 校准数据（50 张 f32 RGB）
│   ├── calibration_data_rgb_f32_320/  # ⚠️ 生成的校准数据（不入库）
│   ├── yolov5s_320_bayese_nv12.yaml   # hb_mapper 量化编译配置
│   └── convert_to_bin.sh           # Docker 一键转换脚本（checker + makertbin）
├── .gitignore
└── README.md
```

---

## 处理流程

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 数据准备     │ →  │ 训练          │ →  │ 导出          │ →  │ 量化          │
│ train_yolo  │    │ yolov5/train │    │ yolov5/export│    │ quantize_    │
│ v5s.py      │    │ .py          │    │ .py          │    │ int8.py      │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  678张图+标注         yolov5s@320px       best.pt            best.onnx
  80/20 划分          300 epochs          → best.onnx        → best_int8.onnx
  → train/val         → best.pt          (26.9MB)            (7.1MB)
                                                    │
                                                    ▼
                                     ┌──────────────────────────┐
                                     │ RDK X5 部署转换           │
                                     │ rdk_x5/ (onnx → .bin)    │
                                     └──────────────────────────┘
```

**详细步骤：**

1. **数据划分**：`train_yolov5s.py` 将 678 张图片按 80/20 随机划分（seed=42），
   复制到 `images/train`、`images/val`（标注同步），并生成 `dataset.yaml`。
2. **训练**：以 COCO 预训练权重 `yolov5s.pt` 为起点，320px、batch=16、300 epochs、patience=0（关闭早停），
   每轮在验证集评估，保存最优权重 `best.pt` 与最后一轮 `last.pt`。
3. **导出**：`export.py` 将 `best.pt` 转为 ONNX 格式 `best.onnx`（含 NMS 后处理）。
4. **量化**：`quantize_int8.py` 从训练集随机抽 200 张（seed=42）作为校准数据，
   使用 ONNX Runtime 静态量化（QDQ、INT8、MinMax、per-channel、仅量化 Conv），
   产出 `best_int8.onnx`，并输出 FP32/INT8 的延迟与输出差异对比。
5. **验证对比**：`val.py` 分别对 FP32 / INT8 ONNX 在验证集评估，结果存 `runs/val/`。

---

## 文件说明

| 文件 | 作用 | 是否必要 |
|---|---|---|
| `iron_steel_dataset_v2/train_yolov5s.py` | 划分数据集 + 启动训练的一键入口 | ✅ 必需 |
| `iron_steel_dataset_v2/dataset.yaml` | 告诉 YOLOv5 数据在哪、有几类 | ✅ 必需 |
| `yolov5/train.py` | 训练主程序 | ✅ 必需 |
| `yolov5/export.py` | .pt → ONNX 导出 | ✅ 必需 |
| `yolov5/val.py` | 验证集评估（mAP/PR） | ✅ 必需 |
| `yolov5/detect.py` | 图片/视频/摄像头推理 | ✅ 必需 |
| `quantize_int8.py` | ONNX INT8 静态量化 + 性能对比 | ✅ 必需（部署） |
| `rdk_x5/export_rdk_onnx.py` | 按地平线规范导出 ONNX（剥离 NMS、NHWC 3 头、batch=1、opset=11） | ✅ 必需（RDK 部署） |
| `rdk_x5/prepare_calib_data.py` | 从训练集生成 hb_mapper 校准数据（50 张 float32 RGB NCHW 0~255） | ✅ 必需（RDK 部署） |
| `rdk_x5/yolov5s_320_bayese_nv12.yaml` | hb_mapper 量化编译配置（march= bayes-e、nv12 输入、data_scale 1/255） | ✅ 必需（RDK 部署） |
| `rdk_x5/convert_to_bin.sh` | Docker 容器内执行 checker + makertbin，产出 .bin | ✅ 必需（RDK 部署） |
| `yolov5/yolov5s.pt` | COCO 预训练权重 | ✅ 必需（训练起点） |
| `iron_steel_dataset_v2/labels/split.py` | 旧划分脚本（路径写死已失效） | ❌ 遗留，可删 |
| `select_calib.py` | 旧校准图选取脚本 | ❌ 遗留，可删 |
| `picture/` | 旧校准图副本 | ❌ 遗留，不入库 |

---

## 运行环境配置

### 基础环境（Windows + conda）

项目在 **conda 环境 `yolov5`** 下运行（Python 3.11.5）：

```bash
# 创建环境（如未创建）
conda create -n yolov5 python=3.11 -y
conda activate yolov5

# 安装 YOLOv5 依赖
cd yolov5
pip install -r requirements.txt

# 核心依赖（已实测版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install onnx onnxruntime onnx-simplifier opencv-python
```

### 实测环境版本

| 组件 | 版本 |
|---|---|
| Python | 3.11.5 |
| PyTorch | 2.8.0+cu129（CUDA 可用） |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.28.0 |
| OpenCV | 4.12.0.88 |
| GPU | NVIDIA RTX 4060 Laptop (8GB) |

> Windows 下训练务必设 `--workers 0`，否则多进程数据加载会崩溃。

---

## 快速开始

### 1. 训练

```bash
# 方式一：一键（自动划分数据 + 训练）
cd iron_steel_dataset_v2
python train_yolov5s.py

# 方式二：手动（数据已划分好）
cd yolov5
python train.py --weights yolov5s.pt ^
    --data "D:\Edge\Elcetronics competition\iron_steel_dataset_v2\dataset.yaml" ^
    --epochs 300 --batch-size 16 --imgsz 320 --patience 0 ^
    --name steel_ball_v1 --cache ram --workers 0
```

### 2. 导出 ONNX

```bash
cd yolov5
python export.py --weights runs/train/steel_ball_v1/weights/best.pt ^
    --imgsz 320 --include onnx
```

### 3. INT8 量化

```bash
# 在项目根目录运行（脚本内路径已配置好）
python quantize_int8.py
```

### 4. 验证

```bash
# FP32
python val.py --data "D:\Edge\Elcetronics competition\iron_steel_dataset_v2\dataset.yaml" ^
    --weights runs/train/steel_ball_v1/weights/best.onnx --imgsz 320 --name fp32_onnx

# INT8
python val.py --data "D:\Edge\Elcetronics competition\iron_steel_dataset_v2\dataset.yaml" ^
    --weights runs/train/steel_ball_v1/weights/best_int8.onnx --imgsz 320 --name int8_onnx
```

### 5. 推理

```bash
python detect.py --weights runs/train/steel_ball_v1/weights/best.pt --source <图片/视频路径> --imgsz 320
```

---

## 训练结果

训练配置：`yolov5s` @ 320px，300 epochs，batch 16，patience 0。

| 指标 | 数值 |
|---|---|
| mAP@0.5 | **0.990** |
| mAP@0.5:0.95 | **0.870** |
| Precision | 0.995 |
| Recall | 0.984 |

### 量化对比（CPU 推理，单张 320px 输入）

| 模型 | 体积 | 说明 |
|---|---|---|
| best.onnx (FP32) | 26.9 MB | 原始精度基准 |
| best_int8.onnx (INT8) | 7.1 MB | 部署用，体积 ↓73% |

> 详细逐 epoch 指标见 `yolov5/runs/train/steel_ball_v1/results.csv`，
> 曲线图见同目录 `*.png`，FP32/INT8 验证集对比见 `runs/val/`。

---

## 部署到 RDK X5

本项目最终目标是在 **地平线 RDK X5**（旭日 X5，BPU）上运行。部署链路：

```
best.pt ──(export_rdk_onnx.py)──> rdk_yolov5s_320.onnx ──(hb_mapper / OpenExplorer 工具链)──> .bin ──> RDK X5 板端 (hobot-dnn)
```

工具链：**OpenExplorer v1.2.8**（Docker 镜像 `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`），`march= bayes-e`。

### 转换步骤（`rdk_x5/` 一键工具链）

```bash
# ① 在 Windows（conda yolov5 环境）导出符合地平线规范的 ONNX
#    - 剥离 NMS 后处理，Detect 头输出 3 个 NHWC 特征图（P3/P4/P5）
#    - 固定 batch=1、opset=11
#    - 输出节点命名 small / medium / big（1×40×40×24 / 1×20×20×24 / 1×10×10×24）
conda activate yolov5
python rdk_x5/export_rdk_onnx.py
# 产物: yolov5/runs/train/steel_ball_v1/weights/rdk_yolov5s_320.onnx

# ② 生成 hb_mapper 校准数据（50 张 float32 RGB NCHW 0~255，不可归一化）
python rdk_x5/prepare_calib_data.py
# 产物: rdk_x5/calibration_data_rgb_f32_320/*.bin

# ③ 在 WSL2/Linux（需 Docker）转换 ONNX → .bin
bash rdk_x5/convert_to_bin.sh
# 产物: rdk_x5/output/yolov5s_320_bayese_nv12/*.bin
```

### 关键配置（`rdk_x5/yolov5s_320_bayese_nv12.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `march` | `bayes-e` | RDK X5 专属架构（X3=bernoulli2，Ultra/J5=bayes） |
| `input_type_rt` | `nv12` | 板端 BPU 最优输入格式 |
| `input_type_train` | `rgb` + `NCHW` | 训练/校准数据格式 |
| `norm_type` / `scale_value` | `data_scale` / `1/255` | YOLOv5 无均值减法，仅缩放 |
| `cal_data_dir` | `calibration_data_rgb_f32_320` | 校准数据目录 |
| `compile_mode` | `latency` | 延时优先编译 |

### 板端要求

- RDK OS ≥ 3.2.3
- 将 `.bin` 拷贝至板端，使用 `hobot_dnn` 加载推理，输入需转为 NV12 格式（或改 yaml 为 `rgb` 直连）

> 参考：[RDK Model Zoo — YOLOv5](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples/vision/yolov5)

---

## 常见问题

**Q: 训练时显存不足？**
A: 降低 batch-size（16 → 8/4）或换 yolov5n。

**Q: Windows 下训练报多进程错误？**
A: 使用 `--workers 0`。

**Q: 量化后模型检测效果变差？**
A: 检查校准图是否来自训练集分布、数量是否充足（≥100）；`quantize_int8.py` 默认 200 张。

**Q: 想换输入尺寸？**
A: 训练 `--imgsz`、导出 `--imgsz`、量化脚本 `INPUT_SIZE`、验证 `--imgsz` 需保持一致（本项目统一 320）。

---

## 许可证

本仓库仅用于 比赛用途，YOLOv5 框架遵循 [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE) 协议，请遵循其开源条款。

---

*数据集与模型权重仅用于竞赛训练与研究，请勿用于商业用途。*