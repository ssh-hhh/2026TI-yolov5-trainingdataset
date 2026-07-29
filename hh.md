但在 X5 上应做这些优化：
1. 优先使用摄像头原生 NV12
X5 模型配置为 input_type_rt: nv12。如果摄像头直接输出 NV12，就不要先转成 BGR，再转 RGB/NV12。直接把 NV12 输入 BPU，可减少颜色转换和内存复制。
2. 统一 letterbox
训练验证、Ubuntu ONNX 推理、校准数据和 X5 推理必须使用同一种 resize 策略。不要一处直接拉伸、一处 letterbox，否则框坐标和量化精度都会变化。
3. 避免重复分配
循环外预分配：
cv::Mat frame;
cv::Mat resized;
std::vector<float> input_buffer;
std::vector<Detection> detections;
循环中使用 clear() 和复用内存，避免频繁创建大对象、clone() 和反复扩容。
4. 三线程流水线
线程1：摄像头采集
线程2：BPU 推理
线程3：后处理、显示、串口
队列长度设为 1 或 2。处理不过来时丢弃旧帧，不要让视频延迟不断累积。
5. 生产环境关闭显示
cv::imshow() 和 waitKey() 会增加 CPU 和显示开销。开发时开启，正式运行时通过参数关闭：
./steel_detector --display=false
6. 降低后处理负担
你的模型只有 3 类，可以：
- 推理后先做置信度过滤
- 再执行 NMS
- 限制最大候选框数量
- 只保留业务需要的类别
- 如果只发送一个目标，NMS 后直接选择最高置信度或最接近画面中心的目标
7. 优先 C++ 部署
官方指出 Python 后处理比 C/C++ 慢。Ubuntu 阶段可以先用 Python验证，最终 X5 建议使用：
- OpenCV C++
- LibDNN 或官方 C++ runtime
- 现有 LibSerial 串口代码
两套前处理
电脑浮点 ONNX 通常接收：
BGR Mat
→ letterbox 320×320
→ BGR 转 RGB
→ float32 / 255
→ NCHW
X5 .bin 推荐接收：
摄像头 NV12
→ letterbox/resize
→ NV12 输入 BPU
两者像素格式不同，但必须共享相同的缩放比例和 padding 信息：
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int original_width;
    int original_height;
};
后处理用它把模型框恢复到原始画面：
x = (x_model - pad_x) / scale;
y = (y_model - pad_y) / scale;
上板验收
在 Ubuntu 和 X5 上用同一批图片，至少比较：
- 检测类别是否一致
- 框坐标误差
- 置信度误差
- Precision、Recall、mAP
- 单帧端到端延迟
- 纯模型推理延迟
- CPU 占用
- 内存占用
- 串口输出是否一致
X5 上先测纯模型：
hrt_model_exec perf \
  --model_file best_x5_320_nv12.bin \
  --thread_num 1 \
  --frame_count 1000
再测完整 OpenCV 应用。两者的差值就是摄像头、预处理、后处理、绘制和串口产生的开销。
官方参考：
- RDK X5 YOLOv5 示例 (https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_x5/samples/vision/yolov5)
- YOLOv5 转换说明 (https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/samples/vision/yolov5/conversion/README_cn.md)
- X5 OpenExplorer 工具链 (https://developer.d-robotics.cc/api/v1/fileData/x5_doc-v126cn/index.html)
最稳妥的实施顺序是：先在 Ubuntu 完成 OpenCV + ONNX Runtime + 串口 的完整 C++ 程序，再按相同接口增加 X5 后端，最后只替换模型加载和推理实现。