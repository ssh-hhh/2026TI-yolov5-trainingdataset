# 将模型部署到 RDK X5 —— 基于 yolov5 系列检测钢珠

基于 **YOLOv5s** 的钢材目标检测，检测三类目标：**铁片 (`ir_sheet`)**、**铁盘 (`ir_disc`)**、**钢珠 (`st_ball`)**。
训练 → 量化 → 转 `.bin` → 地平线 **RDK X5** 板端推理全流程闭环。

---

## 目录

- [两条链路](#两条链路)
- [快速开始](#快速开始)
- [运行环境配置](#运行环境配置)
- [训练结果](#训练结果)
- [模型产物](#模型产物)
- [部署到 RDK X5](#部署到-rdk-x5)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 两条链路

项目从 `best.pt` 出发分两条独立链路，**各自独立量化，互不混用**：

```
                          ┌─────────────┐
                          │   best.pt   │  训练最优权重（唯一源头）
                          └──────┬──────┘
                    ┌────────────┴────────────┐
        链路① 本地   │                         │  链路② 板端部署
        (Windows)   ▼                         ▼   (Windows→Ubuntu)
        export.py              export_rdk_onnx.py (rdk_x5/)
             │                         │
             ▼                         ▼
         best.onnx             rdk_yolov5s_320.onnx
         FP32 26.9MB           FP32 26.8MB（无NMS/NHWC/batch1/opset11）
         (含NMS，可独立推理)         │ ← 位于 rdk_x5/
              │                    ├─ prepare_calib_data.py → 50张校准数据
              ▼                    │   (rdk_x5/calibration_data_rgb_f32_320/)
   quantize_int8.py              ▼
   (ONNX Runtime QDQ)     hb_mapper makertbin (Ubuntu+Docker)
             │              ── 内部完成 INT8 量化 ──
             ▼                    │
        best_int8.onnx            ▼
        INT8 7.1MB         yolov5s_320_....bin (INT8)
        本地PC/边缘验证      ★ RDK X5 板端 hobot_dnn 加载
```

| | 链路① INT8 量化（本地） | 链路② ONNX → .bin（板端部署） |
|---|---|---|
| 输入 | `best.onnx`（FP32，含 NMS） | `rdk_yolov5s_320.onnx`（FP32，RDK 规范） |
| 量化器 | ONNX Runtime（`quantize_int8.py`，QDQ） | **hb_mapper**（地平线工具链，编译时量化） |
| 校准数据 | 训练集随机 200 张（内存读取） | `prepare_calib_data.py` 生成 50 张（0~255 不可归一化） |
| 产物 | `best_int8.onnx`（7.1MB） | `.bin`（INT8 板端模型） |
| 用途 | 本地 PC 端 INT8 精度/速度验证 | RDK X5 板端部署推理 |

> ⚠️ **关键区别**：
> - 链路②的 `.bin` 同样是 INT8 模型，但量化由 **hb_mapper 用校准数据在编译时完成**，
>   输入必须是 **FP32** 的 `rdk_yolov5s_320.onnx`。
> - 链路①的 `best_int8.onnx`（QDQ 格式）**不能**作为链路②的输入——工具链不识别，且会二次量化损失精度。

---

## 快速开始

> ✅ **免配置路径**：所有脚本基于 `__file__` 自动定位仓库根，clone 后无需改路径。

### 1. 训练

```bash
# 一键（自动划分 80/20 数据 + 训练）
cd "train yolov5s"
python train_yolov5s.py

# 或手动（数据已划分）
cd yolov5
python train.py --weights yolov5s.pt --data "../train yolov5s/dataset.yaml" ^
    --epochs 300 --batch-size 16 --imgsz 320 --patience 0 ^
    --name steel_ball_v1 --cache ram --workers 0
```

### 2. 链路①（本地 INT8）

```bash
cd yolov5
python export.py --weights runs/train/steel_ball_v1/weights/best.pt --imgsz 320 --include onnx
# 回到根目录
python quantize_int8.py
# 验证（FP32 / INT8）
python val.py --data "../train yolov5s/dataset.yaml" ^
    --weights runs/train/steel_ball_v1/weights/best.onnx --imgsz 320 --name fp32_onnx
python val.py --data "../train yolov5s/dataset.yaml" ^
    --weights runs/train/steel_ball_v1/weights/best_int8.onnx --imgsz 320 --name int8_onnx
# 推理
python detect.py --weights runs/train/steel_ball_v1/weights/best.pt --source <图片/视频> --imgsz 320
```

### 3. 链路②（转 .bin）

```bash
# Windows（conda yolov5）导出 RDK 规范 ONNX + 校准数据
python rdk_x5/export_rdk_onnx.py
python rdk_x5/prepare_calib_data.py
# 拷贝到 Ubuntu 后转换（详见下方「部署到 RDK X5」）
bash rdk_x5/convert_to_bin.sh
```

---

## 运行环境配置

### Windows + conda（训练/导出/量化，Python 3.11.5）

```bash
conda create -n yolov5 python=3.10 -y
conda activate yolov5
cd yolov5 && pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install onnx onnxruntime onnx-simplifier opencv-python
```

| 组件 | 版本 |
|---|---|
| Python | 3.11.5 |
| PyTorch | 2.8.0+cu129（CUDA 可用） |
| ONNX / ONNX Runtime | 1.22.0 / 1.28.0 |
| OpenCV | 4.12.0.88 |
| GPU | NVIDIA RTX 4060 Laptop (8GB) |

> Windows 训练务必 `--workers 0`。

### Ubuntu（转 .bin）

Docker + 工具链镜像 `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8`，
完整安装步骤见 **[`rdk_x5/ubuntu_deploy_tutorial.md`](rdk_x5/ubuntu_deploy_tutorial.md)**。

---

## 训练结果

`yolov5s` @ 320px，300 epochs，batch 16。

| 指标 | 数值 |
|---|---|
| mAP@0.5 | **0.990** |
| mAP@0.5:0.95 | **0.870** |
| Precision | 0.995 |
| Recall | 0.984 |

本地量化对比（CPU，320px 单张）：

| 模型 | 体积 | 说明 |
|---|---|---|
| best.onnx (FP32) | 26.9 MB | 精度基准 |
| best_int8.onnx (INT8) | 7.1 MB | 本地验证用，体积 ↓73% |

> 逐 epoch 指标见 `yolov5/runs/train/steel_ball_v1/results.csv` 及同目录曲线图。

---

## 模型产物

| 文件 | 来源 | 用途 |
|---|---|---|
| `best.pt` / `last.pt` | 训练 | 源头权重；`best.pt` 为所有导出的输入，`last.pt` 仅用于断点恢复 |
| `best.onnx` | `export.py` | 链路①输入：本地验证 + ONNX Runtime 量化（含 NMS） |
| `best_int8.onnx` | `quantize_int8.py` | 链路①产物：本地 INT8 验证，**不参与** `.bin` 转换 |
| `rdk_yolov5s_320.onnx` | `export_rdk_onnx.py` | 链路②输入（位于 `rdk_x5/`）：RDK 规范 FP32（无 NMS/NHWC/batch=1/opset=11，输出 small/medium/big 三头） |
| `yolov5s_320_bayese_nv12.bin` | `convert_to_bin.sh` | 链路②产物：★ INT8 板端模型，`hobot_dnn` 加载 |

---

## 部署到 RDK X5

工具链：**OpenExplorer v1.2.8**（`march= bayes-e`）。Windows 负责导出/校准，**Ubuntu 完成转换**。

```bash
# ① Windows（conda yolov5）：导出 RDK 规范 ONNX
#    剥离 NMS、NHWC 3 头、固定 batch=1、opset=11
#    输出: small [1,40,40,24] / medium [1,20,20,24] / big [1,10,10,24]
python rdk_x5/export_rdk_onnx.py
#    → rdk_x5/rdk_yolov5s_320.onnx

# ② Windows：生成校准数据（50 张 float32 RGB NCHW 0~255，不可归一化）
python rdk_x5/prepare_calib_data.py
#    → rdk_x5/calibration_data_rgb_f32_320/*.bin

# ③ Ubuntu：拷贝①②产物后转换（checker + makertbin）
bash rdk_x5/convert_to_bin.sh
#    → rdk_x5/output/yolov5s_320_bayese_nv12/*.bin
```

### 关键配置（`rdk_x5/yolov5s_320_bayese_nv12.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `march` | `bayes-e` | RDK X5 专属（X3=bernoulli2，Ultra/J5=bayes） |
| `input_type_rt` | `nv12` | 板端 BPU 最优输入格式 |
| `norm_type` / `scale_value` | `data_scale` / `1/255` | YOLOv5 无均值减法，仅缩放 |
| `cal_data_dir` | `calibration_data_rgb_f32_320` | 校准数据目录 |
| `compile_mode` | `latency` | 延时优先 |

### 板端要求

- RDK OS ≥ 3.2.3；`.bin` 用 `hobot_dnn` 加载，输入转 NV12（或改 yaml `input_type_rt: rgb` 重新转换）
- 板端需自行实现 3 头解码 + NMS（模型不含后处理）

> 参考：[RDK Model Zoo — YOLOv5](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples/vision/yolov5)

---

## 目录结构

```
Elcetronics competition/
├── iron_steel_dataset_v2/          # 数据集（images/labels 各含 train 542 + val 136）
├── train yolov5s/                  # 训练配置与入口
│   ├── dataset.yaml                # 数据集配置（3 类）
│   └── train_yolov5s.py            # 一键：划分 80/20 + 训练
├── yolov5/                         # YOLOv5 框架 + 训练产物
│   ├── runs/train/steel_ball_v1/weights/   # best.pt / last.pt / best.onnx / best_int8.onnx
│   ├── runs/val/                   # FP32/INT8 验证结果
│   ├── train.py / val.py / detect.py / export.py
│   └── yolov5s.pt                  # COCO 预训练权重
├── quantize_int8.py                # 链路① ONNX Runtime 静态 INT8 量化
├── rdk_x5/                         # 链路② 工具链
│   ├── export_rdk_onnx.py          # RDK 规范 ONNX 导出（产物 rdk_yolov5s_320.onnx，可再生成）
│   ├── prepare_calib_data.py       # 校准数据生成（50 张）
│   ├── calibration_data_rgb_f32_320/  # ⚠️ 生成的校准数据（不入库）
│   ├── yolov5s_320_bayese_nv12.yaml# hb_mapper 配置
│   ├── convert_to_bin.sh           # Docker 转换脚本（checker + makertbin）
│   └── ubuntu_deploy_tutorial.md   # Ubuntu 环境教程
├── .gitignore
└── README.md
```

---

## 常见问题

**Q: 转 .bin 报 Unsupported operator / opset 错误？**
A: 必须用 `rdk_x5/export_rdk_onnx.py` 导出的 ONNX（opset=11、batch=1、无 NMS）；`best.onnx` 含 NMS 不支持。

**Q: 板端检测不到目标？**
A: 最常见原因是校准数据被提前归一化（÷255）。`prepare_calib_data.py` 输出 0~255 原始值，`scale_value=1/255` 由工具链处理。

**Q: Ubuntu 拉取工具链镜像慢/失败？**
A: 配置 Docker registry-mirrors，见 `rdk_x5/ubuntu_deploy_tutorial.md` 第 3 节。

**Q: 训练显存不足 / 多进程错误？**
A: 降低 batch-size；Windows 用 `--workers 0`。

**Q: 想换输入尺寸？**
A: 训练 `--imgsz`、导出 `--imgsz`、量化 `INPUT_SIZE`、验证 `--imgsz` 保持一致（本项目 320）。

---

## 许可证

本仓库仅用于 比赛用途，YOLOv5 框架遵循 [AGPL-3.0](https://github.com/ultralytics/yolov5/blob/master/LICENSE) 协议，请遵循其开源条款。

---

*数据集与模型权重仅用于竞赛训练与研究，请勿用于商业用途。*
