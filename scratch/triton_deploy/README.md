# FaceGuard Triton Inference Server

本项目为 WCM (Wasu Content Management) 系统中剥离出的人脸检测与特征提取引擎的微服务版本。基于 NVIDIA Triton Inference Server 的 Python Backend 构建。

本微服务负责在高性能 GPU 环境下，并发接收视频帧图像，利用 `DeepFace` (底层基于 RetinaFace 及多种提取网络) 快速完成人脸侦测与 512 维特征向量提取，并利用 Triton 内置的动态批处理（Dynamic Batching）实现高吞吐。

## 端口说明

为避免与主节点的 API 服务（默认 `8000`）发生端口冲突，本容器对外的端口映射已做平移：

*   **`8001` (映射内部的 8000 端口)**：HTTP / REST API。供 WCM 主服务调用 `v2/models/face_engine/infer` 接口发起推理。
*   **`8002` (映射内部的 8001 端口)**：gRPC API。用于支持后续更高性能（零拷贝、无序列化损耗）的二进制 Tensor 传输。
*   **`8003` (映射内部的 8002 端口)**：Metrics 监控端口。可对接 Prometheus，访问 `http://<ip>:8003/metrics` 查看 GPU 使用率、动态批次大小、队列延迟等健康指标。

## 模型结构与参数

*   **输入**：`IMAGE_BYTES` (1D String Tensor)。接收通过 Base64 编码的 JPEG/PNG 图像字节流。
*   **输出**：
    *   `NUM_FACES` (1D Int32)：画面中检测到的有效人脸数。
    *   `BBOXES` (2D Float32, `[3, 4]`)：最多 3 个人脸的边界框 `[x, y, w, h]`。不足 3 个时将用 0 填充，超出时取占比面积最大的前 3 个。
    *   `EMBEDDINGS` (2D Float32, `[3, 512]`)：最多 3 个人脸的归一化特征向量。

## 部署与启动

进入本目录后，只需执行以下命令即可一键构建并启动容器（需具备 GPU 及 Nvidia-Docker 运行时支持）：

```bash
sudo docker compose up -d --build
```

查看实时日志：
```bash
sudo docker compose logs -f
```

## 客户端配置

在 WCM 主服务的配置文件 `.env` 或 `config.py` 中，指定引擎模式和接口地址：
```python
face_engine_mode = "triton"
triton_server_url = "http://127.0.0.1:8001"
```
