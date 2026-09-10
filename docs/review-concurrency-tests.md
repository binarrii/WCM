# 审核并发测试记录

记录日期：2026-09-10。本文汇总已完成的并发实测和自动化验证；实现机制见[跨进程共享机制](cross-process-sharing.md)，当前配置见[审核并发配置](review-concurrency.md)。

## 1. 当前实现与测试口径

| 层级 | 当前配置或行为 | 验证重点 |
|---|---|---|
| 综合审核任务 | 同一 API 容器的多个进程合计最多 4 个任务，超出后等待名额 | 跨进程限制、排队、异常及取消后释放 |
| 单视频采样窗口 | 默认同时处理 2 个窗口，可配置 | 活跃窗口数量受限，进度与采样点对应 |
| 单窗口审核模块 | face、OCR → guard、visual → guard 并发 | 故障隔离、触发终止后的取消清理 |
| 模型失败保护 | 每个视频、每个模型分别统计最近最多 10 次逻辑调用的最终结果，5 次最终失败即终止视频 | 首轮失败不计数、重试恢复、滚动统计、任务隔离、缓存去重、及时退出 |
| 模型超时与重试 | visual 每轮 50 秒；face、OCR、guard 每轮默认 10 秒；错误或超时最多重试 1 次 | 每轮独立总时限；两轮都失败才计 1 次失败 |

当前综合审核最多可以有 4 × 2 = 8 个活跃窗口。这是调度上限，不代表模型服务经过压测后可稳定承载的容量。

## 2. 失败窗口真实模型对照

### 2.1 原任务

- 任务 ID：`fc955635-a83e-4a86-8ab6-2c231710bb65`。
- 视频：`test_pieces-2.mp4`，时长 589.57 秒。
- 原任务耗时：2838.1 秒，约 47 分 18 秒。
- 共 208 个审核窗口、537 个采样点；127 项 visual 审核未完成，影响 366 个采样点，比例 68.16%。
- 对应日志全部为 visual 描述阶段 `ReadTimeout`；同期 WCM 日志只有该审核任务，最多 4 个活跃窗口。

### 2.2 方法与当时的配置

复测时间为 2026-09-10 11:46～11:51（UTC+8）。从原视频重建 4 组原来失败的窗口，共 10 帧，保持相同 PTS 采样、组窗、模型、提示词及 `max_tokens=300`。每张图片等比缩小至最长边不超过 960 像素，再以 JPEG 质量 90 编码。

分别执行：

1. 四组窗口依次单独调用 visual。
2. 相同四组窗口同时调用 visual。
3. 相同四组窗口同时执行 face、OCR、visual、guard 的完整审核流程。

**这组实测发生在超时配置调整之前。** 串行和纯 visual 并发探针采用 180 秒读取超时，以便观察迟到结果；完整流程采用当时生产的 visual 60 秒读取超时。以下成功结果不能解释为当前每轮 50 秒、最多重试 1 次的 visual 配置已经过实测验证。

### 2.3 实测结果

| 窗口内采样时间（秒） | 串行 visual | 4 并发 visual | 4 并发完整流程中的 visual＋guard |
|---|---:|---:|---:|
| 4、6 | 13.13 秒 | 45.45 秒 | 45.36 秒 |
| 7、10 | 13.02 秒 | 42.82 秒 | 47.03 秒 |
| 35、38、39 | 17.16 秒 | 45.54 秒 | 44.52 秒 |
| 564、565、566 | 13.24 秒 | 43.34 秒 | 46.83 秒 |

8 次纯视觉请求全部返回 HTTP 200、`finish_reason=stop`，输出 45～77 个 token，`reasoning_tokens=0`，没有空响应或截断。4 组完整流程也全部成功，其中 face 为 0.314～0.540 秒，OCR＋guard 为 1.280～1.893 秒。

4 并发传输追踪显示，连接、TLS 和图片上传在 0.036～0.123 秒内完成，响应头约在 42.8～45.5 秒后才返回，正文随后立即完成。主要耗时发生在等待上游返回结果期间。

### 2.4 可以得出的结论与尚未验证的内容

- 抽取的失败输入能够正常处理；本次没有复现超过 60 秒的请求。
- 在这组样本中，并发 4 使单次 visual 耗时明显增加；客户端日志不能进一步区分网关排队、模型推理耗时和其他共享负载。
- 原任务失败的直接原因是读取超时，不能仅凭这些对照认定并发 4 是唯一原因。
- 只复测了 4 组窗口，没有逐一复测全部 127 个失败窗口，也没有完成“4 个真实视频任务同时审核”的模型服务容量测试。
- 后续先将窗口默认并发降为 2、增加 30／10 秒总时限与失败保护，再将 visual 每轮时限改为 50 秒并增加一次重试；本记录不包含这些后续配置下的完整视频性能复测。

原始排查结论与模型返回描述样例见[失败窗口复测与超时提示修复](model-evaluations/2026-09-10-visual-timeouts.md)。

## 3. 自动化并发与故障测试

| 测试文件 | 已验证的行为 |
|---|---|
| [test_review_scheduler.py](../tests/test_review_scheduler.py) | 独立子进程占用锁时，另一进程无法取得同一名额；子进程退出后名额恢复；任务上限 1、4 下排队与释放正确 |
| [test_review_progress.py](../tests/test_review_progress.py) | 窗口并发配置为 1、2、4 时活跃调用受限；乱序完成不会跳过未完成的前置窗口；三图、拼图和回退请求尺寸受限 |
| [test_model_health.py](../tests/test_model_health.py) | 最近 10 次滚动统计、尚未满 10 次已有 5 次失败时提前终止；四模型分别触发；不同视频互不累计；缓存及共用在途请求不重复计数 |
| [test_model_health.py](../tests/test_model_health.py) | window/target 流程终止时取消剩余模块、退出队列等待、记录失败原因、释放任务名额，后续健康任务可以完成；四个实际模型入口均执行总时限 |
| [test_review_stream.py](../tests/test_review_stream.py) | 独立进程间事件传递、初始快照后推送、慢订阅者重同步、客户端断连不取消审核 |
| [test_ifs_adapter.py](../tests/test_ifs_adapter.py) | 人脸服务异常向外抛出供失败保护统计；正常 `face_not_found` 不作为服务失败 |
| [test_review_resilience.py](../tests/test_review_resilience.py) | 未达到终止阈值时单模块故障仍保留其他结果；HTTPX 异常链正确显示超时或连接失败 |

这里的模型调用大多使用可控替身；跨进程锁和事件测试使用真实子进程及本机 IPC。它们验证调度与故障处理正确性，不测量真实模型吞吐。

## 4. 2026-09-10 通用保护初版的验证记录（重试改动之前）

以下两组测试在通用保护与新超时配置实现后已运行，分别 **117 项通过**、**235 项通过**，合计 **352 项通过**。

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_model_health.py tests/test_ifs_adapter.py \
  tests/test_ocr_stream.py tests/test_guard_verdicts.py \
  tests/test_review_resilience.py -q --tb=short

rtk proxy .venv/bin/python -m pytest \
  tests/test_window_review.py tests/test_review_progress.py \
  tests/test_review_scheduler.py tests/test_video_windows.py \
  tests/test_review_tasks.py tests/test_api_routes.py \
  tests/test_review_stream.py tests/test_scene_sampling.py \
  tests/test_nsfw_target_review.py tests/test_face_engine.py -q --tb=short
```

这些命令需要项目 Python 依赖及本机 Unix Socket 权限。上述数字是该日期执行时的记录，后续增加测试可能改变用例数。

当日功能部署后已核验：251 服务健康、前端资源可访问，容器实际配置为任务并发 4、窗口并发 2、visual 超时 30 秒、其余模型超时 10 秒；在新容器中验证四模型均能达到失败阈值并生成正确原因。本次文档整理没有重新发起模型性能测试。

## 5. 2026-09-10 一次重试与 visual 50 秒时限

本次本地改动：四个审核模型错误或超时后最多重试 1 次。首轮失败不占统计窗口；重试成功记 1 次成功，两轮都失败才记 1 次失败。visual 每轮默认 50 秒，其余模型每轮默认 10 秒。OCR 中途网络错误或无完成标志的断流进入重试，避免把半段字幕当作成功结果。

验证覆盖首轮失败后恢复、重试失败后只计一次、每轮重新计时、四个实际入口最多尝试两轮、共用请求不重复计数、guard 重试不重复调用成功的 visual，以及终止后取消在途请求并释放任务名额。window/target 两种审核模式均验证。

```bash
rtk proxy .venv/bin/python -m pytest \
  tests/test_model_health.py tests/test_ocr_stream.py \
  tests/test_guard_verdicts.py tests/test_review_resilience.py \
  tests/test_nsfw_target_review.py -q --tb=short

rtk proxy .venv/bin/python -m pytest \
  tests/test_ifs_adapter.py tests/test_analyze_media_faces.py \
  tests/test_window_review.py tests/test_review_progress.py \
  tests/test_review_scheduler.py tests/test_video_windows.py \
  tests/test_scene_sampling.py tests/test_face_engine.py -q --tb=short
```

第一组 **133 项通过**；第二组 **192 项通过**，另 4 项 API 测试首次因沙箱禁止 Unix Socket 绑定而无法启动，经允许在沙箱外单独重跑后 **4 项通过**，合计 **329 项通过**。Ruff 检查和 `git diff --check` 通过。

本次没有连接真实模型进行性能复测，也没有同步或重启 251；上一节的线上配置核验是重试改动之前的历史记录。
