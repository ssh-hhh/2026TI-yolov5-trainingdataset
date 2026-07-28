# -*- coding: utf-8 -*-
"""钢珠实时检测 - CPU模式（适配边缘计算平台）"""
import sys, os, time, cv2, torch
import numpy as np

# === 配置 ===
MODEL_PATH = r"D:\Edge\Elcetronics competition\yolov5\runs\train\steel_ball_v1\weights\best.pt"
IMG_SIZE = 640          # CPU 上 640px 够用，速度翻倍
CONF_THRES = 0.5
IOU_THRES  = 0.45
CAMERA_ID  = 0

# === 加载模型（CPU 模式，边缘设备无 GPU 加速） ===
yolov5_dir = r"D:\Edge\Elcetronics competition\yolov5"
os.chdir(yolov5_dir)
sys.path.insert(0, yolov5_dir)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

device = torch.device("cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names = model.stride, model.names
model.warmup(imgsz=(1, 3, IMG_SIZE, IMG_SIZE))

print(f"模型加载完成 | CPU 模式 | 类别: {names}")

# === 摄像头 ===
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("摄像头已开启，按 Q 退出")

fps = 0.0
frame_count = 0
t_start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # 预处理
    img = letterbox(frame, IMG_SIZE, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    # 推理
    pred = model(img)
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, max_det=100)

    t_infer = (time.time() - t0) * 1000

    # 画框
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ball {conf:.2f}", (x1 + 2, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 帧率
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10.0 / (time.time() - t_start)
        t_start = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f} | Infer: {t_infer:.0f}ms | Thresh: {CONF_THRES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Detections: {len(pred[0]) if len(pred) else 0}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Steel Ball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
