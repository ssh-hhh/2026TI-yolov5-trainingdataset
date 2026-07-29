"""Create an ONNX Runtime QDQ INT8 model from the trained YOLOv5s model."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT: Final = Path(r"D:\Edge\Elcetronics competition")
MODEL_PATH: Final = ROOT / "yolov5/runs/train/steel_ball_v1/weights/best.onnx"
OUTPUT_PATH: Final = ROOT / "yolov5/runs/train/steel_ball_v1/weights/best_int8.onnx"
CALIBRATION_DIR: Final = ROOT / "iron_steel_dataset_v2/images/train"
INPUT_SIZE: Final = 320
CALIBRATION_COUNT: Final = 200
RANDOM_SEED: Final = 42


def letterbox(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Resize an image with centered padding to match YOLOv5 inference."""
    height, width = image.shape[:2]
    ratio = min(INPUT_SIZE / height, INPUT_SIZE / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    pad_width = INPUT_SIZE - resized_width
    pad_height = INPUT_SIZE - resized_height
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def preprocess(image_path: Path) -> npt.NDArray[np.float32]:
    """Load and transform one calibration image into NCHW RGB float32."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        msg = f"Cannot read calibration image: {image_path}"
        raise FileNotFoundError(msg)
    image = cv2.cvtColor(letterbox(image), cv2.COLOR_BGR2RGB)
    tensor = image.astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


class YoloCalibrationReader(CalibrationDataReader):
    """Feed deterministic representative images to ONNX Runtime calibration."""

    def __init__(self, input_name: str, image_paths: list[Path]) -> None:
        self._input_name = input_name
        self._iterator: Iterator[Path] = iter(image_paths)

    def get_next(self) -> dict[str, npt.NDArray[np.float32]] | None:
        image_path = next(self._iterator, None)
        if image_path is None:
            return None
        return {self._input_name: preprocess(image_path)}


def select_calibration_images() -> list[Path]:
    """Select a reproducible, non-sequential calibration subset."""
    images = sorted(CALIBRATION_DIR.glob("*.jpg"))
    if len(images) < CALIBRATION_COUNT:
        msg = f"Need {CALIBRATION_COUNT} images, found {len(images)} in {CALIBRATION_DIR}"
        raise RuntimeError(msg)
    return random.Random(RANDOM_SEED).sample(images, CALIBRATION_COUNT)


def benchmark(session: ort.InferenceSession, tensor: npt.NDArray[np.float32]) -> tuple[float, npt.NDArray[np.float32]]:
    """Return median inference latency and first output for one input tensor."""
    input_name = session.get_inputs()[0].name
    for _ in range(5):
        session.run(None, {input_name: tensor})
    durations: list[float] = []
    output: npt.NDArray[np.float32] | None = None
    for _ in range(30):
        started = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        durations.append((time.perf_counter() - started) * 1_000)
        output = outputs[0]
    if output is None:
        msg = "ONNX Runtime returned no output"
        raise RuntimeError(msg)
    return float(np.median(durations)), output


def main() -> None:
    """Quantize the current FP32 model and verify loadability and CPU latency."""
    model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(model)
    input_name = model.graph.input[0].name
    calibration_images = select_calibration_images()

    quantize_static(
        model_input=MODEL_PATH,
        model_output=OUTPUT_PATH,
        calibration_data_reader=YoloCalibrationReader(input_name, calibration_images),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        op_types_to_quantize=["Conv"],
    )

    quantized_model = onnx.load(OUTPUT_PATH)
    onnx.checker.check_model(quantized_model)
    providers = ["CPUExecutionProvider"]
    fp32_session = ort.InferenceSession(MODEL_PATH, providers=providers)
    int8_session = ort.InferenceSession(OUTPUT_PATH, providers=providers)
    tensor = preprocess(calibration_images[0])
    fp32_latency, fp32_output = benchmark(fp32_session, tensor)
    int8_latency, int8_output = benchmark(int8_session, tensor)
    difference = np.abs(fp32_output - int8_output)

    print(f"Calibration images: {len(calibration_images)}")
    print(f"FP32 size: {MODEL_PATH.stat().st_size / 1_048_576:.2f} MiB")
    print(f"INT8 size: {OUTPUT_PATH.stat().st_size / 1_048_576:.2f} MiB")
    print(f"FP32 median latency: {fp32_latency:.2f} ms")
    print(f"INT8 median latency: {int8_latency:.2f} ms")
    print(f"Output mean absolute difference: {difference.mean():.6f}")
    print(f"Output maximum absolute difference: {difference.max():.6f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
