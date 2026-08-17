#!/usr/bin/env bash
# ============================================================
# RDK X5 YOLOv5s@320 ONNX -> .bin 一键转换
# 前置条件: Ubuntu + Docker (镜像 openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8)
#
# 用法:
#   1) 在 Windows 上先导出 ONNX 并生成校准数据:
#        conda activate yolov5
#        python rdk_x5/export_rdk_onnx.py
#        python rdk_x5/prepare_calib_data.py
#   2) 将 rdk_x5/ 目录拷到 Ubuntu 后运行本脚本:
#        bash rdk_x5/convert_to_bin.sh
# ============================================================
set -e

# 仓库根目录 (脚本所在目录的上一级)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RDK_DIR="$ROOT_DIR/rdk_x5"
VERSION="v1.2.8"
IMAGE="openexplorer/ai_toolchain_ubuntu_20_x5_cpu:${VERSION}"
ONNX="$RDK_DIR/rdk_yolov5s_320.onnx"

echo "=============================================="
echo " RDK X5 模型转换 (${VERSION})"
echo " ONNX: $ONNX"
echo "=============================================="

# 0. 检查文件
if [ ! -f "$ONNX" ]; then
    echo "[错误] 未找到 $ONNX, 请先运行 export_rdk_onnx.py"
    exit 1
fi
if [ ! -d "$RDK_DIR/calibration_data_rgb_f32_320" ]; then
    echo "[错误] 未找到校准数据目录, 请先运行 prepare_calib_data.py"
    exit 1
fi

# 1. 拉取工具链镜像 (若已存在则跳过)
echo "[1/4] 检查 Docker 镜像..."
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

# 2. 启动容器并执行模型检查
echo "[2/4] hb_mapper checker 模型检查..."
docker run --rm -v "$ROOT_DIR":/data -w /data/rdk_x5 "$IMAGE" \
    hb_mapper checker --model-type onnx --march bayes-e \
    --model "/data/rdk_x5/rdk_yolov5s_320.onnx"

# 3. 量化 + 编译 -> .bin
echo "[3/4] hb_mapper makertbin 量化编译..."
docker run --rm -v "$ROOT_DIR":/data -w /data/rdk_x5 "$IMAGE" \
    hb_mapper makertbin --model-type onnx \
    --config yolov5s_320_bayese_nv12.yaml

# 4. 输出产物
echo "[4/4] 转换完成, 产物:"
find "$RDK_DIR/output" -name "*.bin" -exec ls -lh {} \; 2>/dev/null || true
echo "提示: 复制 .bin 到 RDK X5 板端, 用 hobot_dnn 加载推理"
