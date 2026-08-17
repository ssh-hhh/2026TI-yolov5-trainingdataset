# Ubuntu 环境部署教程：ONNX → RDK X5 (.bin)

> 适用场景：Windows 上用 conda 完成 **训练 / 导出 ONNX / 生成校准数据**（保持原样），
> 将产物拷到 **Ubuntu** 上运行 `hb_mapper` 工具链，把 ONNX 转成 RDK X5 可加载的 `.bin`。

---

## 1. 部署架构

```
┌──────────────────────── Windows ────────────────────────┐
│ conda yolov5 环境（Python 3.11.5）                      │
│ ① python rdk_x5/export_rdk_onnx.py     → rdk_yolov5s_320.onnx │
│ ② python rdk_x5/prepare_calib_data.py  → calibration_data_rgb_f32_320/ │
└──────────────────────────┬──────────────────────────────┘
                           │ 拷贝（U盘 / scp）
                           ▼
┌────────────────────────── Ubuntu ───────────────────────┐
│ Docker + openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8 │
│ ③ bash rdk_x5/convert_to_bin.sh        → *.bin          │
└──────────────────────────┬──────────────────────────────┘
                           ▼
                   RDK X5 板端（hobot-dnn 加载推理）
```

---

## 2. 前置条件

| 项目 | 要求 |
|---|---|
| 系统 | Ubuntu 20.04 / 22.04（x86_64） |
| Docker | ≥ 20.10（详见下文安装） |
| 磁盘 | 工具链镜像约 4~5 GB，预留 ≥ 20 GB |
| 内存 | ≥ 8 GB（推荐 16 GB） |
| 网络 | 能访问 Docker Hub（国内可配镜像加速） |

---

## 3. 安装 Docker

```bash
# ① 安装（使用官方脚本，或 apt 安装 docker.io）
curl -fsSL https://get.docker.com | sudo sh

# ② 将当前用户加入 docker 组（免 sudo 运行），然后重新登录终端
sudo usermod -aG docker $USER
newgrp docker

# ③ 验证
docker --version
docker run hello-world   # 能打印 Hello from Docker! 即成功
```

> 国内 Docker Hub 拉取慢/失败时，配置镜像加速器：
> 编辑 `/etc/docker/daemon.json`：
> ```json
> {
>   "registry-mirrors": [
>     "https://docker.m.daocloud.io",
>     "https://dockerproxy.com",
>     "https://docker.1ms.run"
>   ]
> }
> ```
> 然后 `sudo systemctl restart docker`。

---

## 4. 从 Windows 拷贝所需文件

在 Windows 上先确保①②已执行（在 `D:\Edge\Elcetronics competition` 下）：

```bash
conda activate yolov5
python rdk_x5/export_rdk_onnx.py
python rdk_x5/prepare_calib_data.py
```

然后拷到 Ubuntu（示例目标目录 `~/steel_yolov5`）：

```bash
# 方式一：scp（Windows PowerShell）
scp -r "D:\Edge\Elcetronics competition" user@ubuntu_ip:~/steel_yolov5

# 方式二：U 盘手动拷贝
```

**转换必需的最小文件集**（其余可省略）：

```
steel_yolov5/
├── rdk_x5/
│   ├── convert_to_bin.sh                    # 转换脚本
│   ├── yolov5s_320_bayese_nv12.yaml         # hb_mapper 配置
│   └── calibration_data_rgb_f32_320/        # 校准数据（50 个 .bin）
└── yolov5/runs/train/steel_ball_v1/weights/
    └── rdk_yolov5s_320.onnx                 # 按 RDK 规范导出的 ONNX
```

---

## 5. 拉取工具链镜像

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8
```

> 镜像约 4~5 GB，首次拉取较久。国内网络失败请先配置第 3 节的镜像加速。

---

## 6. 运行转换

```bash
cd ~/steel_yolov5
bash rdk_x5/convert_to_bin.sh
```

脚本会自动执行三步：

1. 检查镜像是否存在（不存在自动拉取）
2. `hb_mapper checker` —— 检查 ONNX 是否符合工具链要求（报错则按提示修正）
3. `hb_mapper makertbin` —— 量化 + 编译，产出 `.bin`

---

## 7. 产物与验证

```bash
# 转换产物目录
ls -lh rdk_x5/output/yolov5s_320_bayese_nv12/

# 应看到
yolov5s_320_bayese_nv12.bin        # 部署用模型（板端加载这个）
yolov5s_320_bayese_nv12.html       # 网络结构可视化
yolov5s_320_bayese_nv12.json       # 模型信息
yolov5s_320_bayese_nv12_quant_info.json  # 量化信息
```

**精度验证（可选，推荐）**：`hb_mapper` 会输出量化前后各层的余弦相似度；
若整体相似度 < 0.98 或检测效果明显下降，检查校准数据是否与真实部署场景分布一致。

---

## 8. 常见问题

**Q: `docker: permission denied`？**
A: 执行 `sudo usermod -aG docker $USER` 后**重新登录终端**，或临时用 `sudo docker ...`。

**Q: 镜像拉取超时/失败？**
A: 按第 3 节配置 registry-mirrors 后 `sudo systemctl restart docker` 重试。

**Q: checker 报 `Unsupported operator` / opset 错误？**
A: 确认 ONNX 是 `export_rdk_onnx.py` 导出的（opset=11、batch=1、无 NMS）。
   旧的 `best.onnx`（含 NMS 后处理）工具链不支持，必须用 RDK 规范导出版。

**Q: 转换后检测不到目标？**
A: 最常见原因是校准数据被提前归一化（除以 255）。确认 `prepare_calib_data.py`
   输出为 0~255 原始值，yaml 中 `scale_value: 0.003921568627451`（=1/255）只由工具链处理。

**Q: 板端加载 .bin 报格式错误？**
A: 确认 march 匹配：RDK X5 用 `bayes-e`；若目标是 RDK X3（bernoulli2）或 Ultra（bayes），
   需修改 yaml 中 `march` 并重新转换。

---

## 9. 板端部署（RDK X5）

1. 将 `.bin` 拷贝到板端（`scp` 或 U 盘）
2. RDK OS ≥ 3.2.3
3. 使用 `hobot_dnn` 加载推理，输入需为 **NV12** 格式（与 yaml 中 `input_type_rt: nv12` 对应）；
   若板端使用 RGB 输入，改 yaml `input_type_rt: rgb` 后重新转换

> 参考：[地平线 RDK 文档](https://developer.d-robotics.cc/) · [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo)