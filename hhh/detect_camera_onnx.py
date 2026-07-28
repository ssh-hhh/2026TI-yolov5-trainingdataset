import sys, os, time, cv2
import numpy as np
import onnxruntime as ort

# === 配置 ===
MODEL_PATH = r"D:\Edge\Elcetronics competition\yolov5\runs\train\steel_ball_v1\weights\best.onnx"
IMG_SIZE = 640
CONF_THRES = 0.55
IOU_THRES  = 0.45
CAMERA_ID  = 0

# === 加载 ONNX 模型 ===
ort.set_default_logger_severity(3)
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"ONNX 模型加载完成 | 输入: {session.get_inputs()[0].shape} | CPU 模式")

# === 摄像头 ===
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"摄像头: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}，按 Q 退出")

fps = 0.0
frame_count = 0
fps_timer = time.time()

# === 预处理 + NMS ===
def preprocess(frame):
    h, w = frame.shape[:2]
    r = IMG_SIZE / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img = cv2.resize(frame, (new_w, new_h))
    pad_w = IMG_SIZE - new_w
    pad_h = IMG_SIZE - new_h
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=114)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32), r, (pad_w, pad_h)

def nms_boxes(boxes, scores, iou_thres):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
                          (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1]) - w * h)
        order = order[1:][iou <= iou_thres]
    return keep

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()

    # 预处理 → 固定输出 IMG_SIZE×IMG_SIZE
    img, ratio, _ = preprocess(frame)
    assert img.shape[2] == IMG_SIZE and img.shape[3] == IMG_SIZE, f"尺寸错: {img.shape}"

    # ONNX 推理
    outputs = session.run(None, {"images": img})
    pred = outputs[0]  # shape: (1, 25200, 6) for YOLOv5n

    # 后处理
    pred = pred[0]  # remove batch dim
    conf_mask = pred[:, 4] >= CONF_THRES
    pred = pred[conf_mask]

    boxes_list = []
    if len(pred):
        # 解码: cx cy w h -> x1 y1 x2 y2（pad只在右下，直接除以ratio还原）
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - w / 2) / ratio
        y1 = (cy - h / 2) / ratio
        x2 = (cx + w / 2) / ratio
        y2 = (cy + h / 2) / ratio

        scores = pred[:, 4]
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms_boxes(boxes, scores, IOU_THRES)
        boxes, scores = boxes[keep], scores[keep]

        for (bx1, by1, bx2, by2), sc in zip(boxes, scores):
            ix1, iy1 = max(0, int(bx1)), max(0, int(by1))
            ix2, iy2 = min(frame.shape[1], int(bx2)), min(frame.shape[0], int(by2))
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {sc:.2f}", (ix1 + 2, iy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            boxes_list.append((ix1, iy1, ix2, iy2, sc))

    t_total = (time.time() - t0) * 1000

    # FPS
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10.0 / (time.time() - fps_timer)
        fps_timer = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f} | Total: {t_total:.0f}ms | Det: {len(boxes_list)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Steel Ball Detection (ONNX)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
