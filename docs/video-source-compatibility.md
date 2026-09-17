# 视频接入与复核媒体

HTTP(S) 视频审核支持 TS、常规视频文件和有限的 HLS 点播。视频地址可带查询参数、经过重定向或没有扩展名。未知扩展名通过有界 GET 判断内容；下载后使用 FFprobe 探测，不将未知二进制当作图片解码。

HLS 支持主/媒体播放列表、相对 URI、TS/fMP4 分片、初始化段、字节范围、独立音轨以及 AES-128。默认选取不超过 1080p 的最高档；若所有档位都高于该值则使用现有最高档。选择结果保存在任务媒体信息中。地址按标准 URI 规则解析，父列表的查询参数不会擅自附加给分片。没有 ENDLIST 的直播/更新中列表、缺失分片、DRM/SAMPLE-AES 会明确拒绝；持续直播与限时录制尚未开放。

## 审核与播放一致性

下载与分片获取均流式落盘。FFmpeg 仅访问已下载的本地文件；远端播放列表不会直接交给 FFmpeg。兼容 H.264/yuv420p 视频复制编码，其他视频转 H.264；AAC 可直接复制，其他音频转 AAC。输出 MP4 写入 faststart 索引。所有审核入口复用这份文件。

OpenCV 的采样时钟相对第一视频帧，浏览器的媒体时钟可能包含音频前导。因此记录输出视频轨道的 `video_start_seconds`，播放器跳转时加该偏移，帧回调与时间轴读数减该偏移。不能把音频与视频起始时间当成相同。验收同时检查逐帧画面和 FFprobe PTS。

任务媒体保存到私有 S3 的独立 `wcm/review-media` 前缀，原有人物图片命名空间及元数据不变。接口为 `/api/v1/review_tasks/{task_id}/media/{media_id}`，要求 `review.read` 权限，支持 GET、HEAD、Range、ETag、If-Range。只允许当前任务绑定的不可变媒体版本，不接受任意对象 Key 或代理 URL。

媒体在执行租约校验后发布。失败重试开始时清除上一版本绑定；未发布对象仍有数据库登记，可回收。下载/FFmpeg 取消会关闭连接、终止并等待子进程、清理临时目录；S3 分片上传异常会执行 abort。上传进程被强制杀死时，运维仍应为此视频前缀配置未完成 multipart 的清理策略。

复核媒体默认保留 7 天，Worker 每分钟清理一批过期、已删除任务及超过一天的未引用媒体；执行中的任务不会被清理。到期播放返回 410，保留审核结果。历史任务仍保留原结果；无法直接播放的旧 TS/HLS 需重新提交以生成媒体，导入 JSON 不会自动下载/转换视频。

## 参数与限制

参数配置页的“视频接入”分组可编辑以下值。`mb` 使用 MiB（1024² 字节）。

| 参数 | 默认 | 含义 |
|---|---:|---|
| max_video_size_mb | 10000 | 普通文件实际读取字节；HLS 清单、密钥、所选音视频分片累计字节 |
| max_video_output_mb | 20000 | 输出复核文件上限 |
| max_video_duration_seconds | 43200 | 最长媒体时长，12 小时 |
| video_prepare_timeout_seconds | 7200 | 单次下载、合并、转换的总时限 |
| video_prepare_concurrency | 2 | 集群同时转换的任务数 |
| video_min_free_disk_mb | 2048 | 临时磁盘必须保留的空闲空间 |
| video_retention_days | 7 | 复核视频保留天数 |
| hls_max_height | 1080 | 首选 HLS 清晰度上限 |
| hls_max_segments | 50000 | 每条所选媒体列表的最大分片数 |

首次初始化独立视频上限时，从已有 `max_file_size_mb × 100` 迁移；之后两个参数独立生效。图片仍使用 `max_file_size_mb`。任务参数沿用原有提交快照机制。超限、无效媒体、直播等永久错误不自动重试；源站临时错误保留重试机制。Nginx 的上传请求体限制不控制服务器拉取 URL 的大小。

## 部署和回滚

新增 nullable `review_tasks.media` 与 `review_media` 表；初始化幂等，旧任务不要求回填。先核对服务器独立修改并保存源文件、镜像和 MySQL 快照，构建成功后在无在途任务时更新 api、worker、webui。旧版 8001、InsightFace、SQLite 和人物 metadata 无需修改。

回滚可恢复本次备份的文件和原镜像，新增表/列保留即可，无需删除审核数据。回滚前等待新格式任务结束或由用户取消，避免旧 Worker 接手 HLS/TS 任务。

验证：`pytest tests/test_media_source.py tests/test_review_media.py`，前端 `npm test && npm run build`。真实存储验收使用 `python -m scripts.verify_review_media --source-url URL`，必须配置新建 `wcm_verify_*` 数据库及 `WCM_REVIEW_MEDIA_PREFIX=wcm/verify-media/...`。该脚本使用真实 FFmpeg/MySQL/S3、独立进程读取播放资源，不调用审核模型或修改人物库。
