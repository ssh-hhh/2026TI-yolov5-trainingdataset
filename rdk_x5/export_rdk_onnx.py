"""
导出符合地平线 RDK X5 (OpenExplorer v1.2.8, march= bayes-e) 规范的 ONNX 模型。

参考: D-Robotics/rdk_model_zoo 的 yolov5 转换说明
关键点:
  1. Detect 头输出改为 NHWC (P3/P4/P5 三个特征图), 剥离 NMS/解码后处理
  2. 固定 batch=1 (无动态轴)
  3. opset=11 (工具链支持 opset 10/11, 更高版本会报不支持)
  4. 输出节点命名: small / medium / big

本脚本通过 monkey-patch Detect.forward 实现, 不修改 yolov5 源码,
不影响训练/推理代码。
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT: Path = Path(__file__).resolve().parent.parent  # 仓库根（脚本位于 rdk_x5/，clone 后无需改路径）
YOLOV5_DIR: Path = ROOT / "yolov5"
WEIGHTS: Path = YOLOV5_DIR / "runs/train/steel_ball_v1/weights/best.pt"
OUTPUT: Path = YOLOV5_DIR / "runs/train/steel_ball_v1/weights/rdk_yolov5s_320.onnx"
INPUT_SIZE: int = 320
OPSET: int = 11

import sys

sys.path.insert(0, str(YOLOV5_DIR))

from models.experimental import attempt_load  # noqa: E402
from models.yolo import Detect  # noqa: E402


def _rdk_forward(self: Detect, x: list[torch.Tensor]) -> list[torch.Tensor]:
    """地平线规范的 Detect 头: 输出 3 个 NHWC 特征图 (无 NMS/解码)。"""
    return [self.m[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]


def main() -> None:
    # 1. 加载已训练权重 (CPU; 旧版 yolov5 attempt_load 内部已 map_location='cpu')
    model = attempt_load(str(WEIGHTS))
    model.eval()

    # 2. 替换 Detect.forward 为 RDK 导出版本 (仅影响导出, 原代码不受影响)
    Detect.forward = _rdk_forward

    # 3. 固定 batch=1 的 dummy 输入
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

    # 4. 导出 ONNX
    torch.onnx.export(
        model,
        dummy,
        str(OUTPUT),
        opset_version=OPSET,
        input_names=["images"],
        output_names=["small", "medium", "big"],
        dynamic_axes=None,  # 固定 batch=1, 禁止动态轴
        do_constant_folding=True,
        verbose=False,
    )
    print(f"导出成功: {OUTPUT}")

    # 5. 可选: onnxsim 简化
    try:
        import onnx
        import onnxsim

        model_onnx = onnx.load(str(OUTPUT))
        simplified, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(simplified, str(OUTPUT))
            print(f"onnxsim 简化完成: {OUTPUT}")
        else:
            print("onnxsim 简化失败 (忽略, 使用原模型)")
    except ImportError:
        print("onnxsim 未安装, 跳过简化 (pip install onnxsim)")

    # 6. 输出节点形状检查
    import onnx

    m = onnx.load(str(OUTPUT))
    print("输出节点:")
    for out in m.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")


if __name__ == "__main__":
    main()
