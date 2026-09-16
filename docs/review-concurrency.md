# 审核并发配置

相关文档：[审核并发测试记录](review-concurrency-tests.md)、[跨进程共享机制](cross-process-sharing.md)。

在 WebUI 的“参数配置”页面修改以下运行时参数，保存后即可生效：

```text
review_task_concurrency=4
review_window_concurrency=4
face_neighbor_concurrency=4
insightface_concurrency=32
ocr_concurrency=32
visual_concurrency=6
guard_concurrency=24
jpeg_quality=95
```

以上是首次初始化默认值；更新程序不会覆盖数据库中已经保存的参数值。任务并发和窗口并发默认均为 4，必须为正整数。JPEG 压缩质量默认 95，有效范围为 1～100。

- 任务上限覆盖 `/analyze_media` 的 HTTP 与 WebSocket 入口，从下载到结果保存占用一个名额。超出上限的任务显示“排队中”，等待空闲名额后开始下载。
- 集群模式使用 MySQL 任务队列统一控制任务名额，API 和 Worker 副本共享上限；任务由独立 Worker 执行，API 重启不会取消任务。单机模式使用容器临时目录中的文件锁共享名额，不随 Gunicorn 进程数倍增。
- 窗口上限按每个视频任务计算，`window` 和 `target` 两种综合审核模式都使用此配置。每个窗口内仍并发执行 face、OCR → guard、visual → guard。
- 默认最多同时审核 4 个任务、16 个窗口。模型服务自身容量仍会影响实际吞吐。

单项 `/detect_sensitive`、`/detect_nsfw` 和人脸搜索接口不占综合审核任务名额。251 当前采用集群部署，生命周期与租约恢复见[集群部署](cluster-deployment.md)。

## 可选模型并发限制

`insightface_concurrency`、`ocr_concurrency`、`visual_concurrency`、`guard_concurrency` 为正整数时，在集群内分别限制对应模型的调用并发。设为 **≤ 0 的整数**时跳过该模型的集群名额申请，由模型服务端控制并发；不会关闭人物写锁、超时、重试或失败保护。单机模式本来就不使用这一层集群模型限流。

`face_neighbor_concurrency` 控制单个视频的补帧识别并发，正数范围 1～8，≤ 0 不限制。补帧总量仍受每窗口与全视频补帧预算约束。如果 `insightface_concurrency` 仍为正数，补帧与其他人脸调用继续共享该集群上限。

`review_task_concurrency`、`review_window_concurrency` 保持必须 ≥ 1，Worker 本地任务容量也保持原约束。模型全局限额读取实时配置；窗口和补帧参数使用提交任务时保存的参数快照，因此应在新提交任务上验证修改效果。

## 模型服务异常提前终止

`api/model_health.py` 提供可复用的 `model_call(model)` 装饰器和 `call_model(model, operation)`。visual、OCR、guard、face 调用错误或超时后立即重试 1 次，最多执行两轮；首轮失败不计入统计，重试成功只记 1 次成功，两轮都失败才记 1 次失败。各模型维护最近最多 10 次逻辑调用的最终结果，每个视频独立统计。任一模型最终失败累计达到 5 次，立即终止该视频的整个审核任务。前 10 次尚未完成但已有 5 次失败时也触发；统计按完成顺序滚动，不按互不重叠的十次分组。

- 缓存命中、共用同一个在途请求的其他等待者不重复计数。OCR 纯色跳过、guard 空文本跳过不计数。
- 自动多图兼容回退属于同一次视觉描述操作，以最终结果计数；空响应或其他无效结果属于错误，正常空 OCR / 未检测到人脸属于成功。
- 重试覆盖完整逻辑模型操作；视觉兼容回退及可选目标帧复核可能包含多次 HTTP 请求。guard 失败只重试 guard，不重新调用已成功的 OCR 或 visual。OCR 流式网络错误、无完成标志的断流也重试，不将半段文字当作成功；正常达到 token / 字数上限或重复检测的可用输出仍按原规则处理。
- guard 在 OCR 和 visual 分支中的调用共用该视频的 guard 统计，但不会计入 OCR 或 visual 的失败次数。取消及由其他模型触发的终止也不重复计为失败。
- face 搜索只对 `face_not_found` 视为无可用人脸；上游服务与传输错误向外抛出，不能当作正常空结果吞掉。
- 触发后停止采样和排队，取消该视频在途审核协程及模型请求，释放任务名额；其他视频不受影响。
- 任务标记 failed，页面仅显示简短原因，例如“任务执行失败：visual 视觉模型调用频繁错误或超时，已提前终止该审核任务。”；其他模型使用对应名称。近期调用数、重试后失败数及阈值保留在服务日志中。提前终止不会将未审核片段标为通过，也不会保存为完整审核结果。

保护覆盖综合审核和独立文字/视觉审核的视频流程，包括 window/target 模式。单图片审核调用同样使用重试及超时限制，但不累计跨图片的失败次数。取消及已触发的视频终止不重试。

## 调用超时

| 模型 | 每轮默认总时限 | 参数键 |
|---|---:|---|
| visual | 50 秒 | `visual_timeout_s` |
| OCR | 10 秒 | `ocr_timeout_s` |
| guard | 10 秒 | `guard_timeout_s` |
| face | 10 秒 | `insightface_timeout_s` |

HTTP 客户端超时及每轮异步总时限均使用这些配置，保存后当前 worker
立即刷新，其他 worker 最迟约 2 秒同步。重试重新获得完整时限，visual
两轮的时限预算合计最多 100 秒，其余模型合计最多 20 秒（另有取消清理开销）。每轮总时限包含整个逻辑模型调用（OCR 流式读取、视觉兼容回退、人脸检索及结果补充），不会因持续返回少量数据而无限延长。视频下载使用原有独立超时。

客户端取消不能保证上游推理服务立即停止已经接收的计算；face 的同步 SDK 在线程中运行，受底层 10 秒网络超时约束，取消等待无法强制终止 Python 线程。
