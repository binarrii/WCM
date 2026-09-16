# WCM 多实例开发部署

本轮实现最初方案的第 1–3 项。InsightFace Server、其模型、SQLite 和索引部署保持原状。

后续新增的独立 InsightFace 副本、可靠同步及故障恢复流程见 [InsightFace 可靠副本同步](insightface-replication.md)。
原主节点保持原镜像和数据格式，新增副本不与原主节点共用 SQLite 文件。

## 251 上的两套入口

| 项目 | 旧版备用 | 新架构 |
|---|---|---|
| 地址 | http://10.252.25.251:8001 | http://10.252.25.251:8000 |
| 目录 | /home/aigc/wcm | /home/aigc/wcm-cluster |
| Compose 项目 | wcm | wcm-cluster |
| API | wcm-facerec-api，原镜像 | api，默认 2 个副本 |
| WebUI | 原镜像内置 | 独立 WebUI/Nginx 镜像 |
| 审核执行 | API 内执行 | worker，默认 2 个副本，每个 2 个任务 |
| MySQL | 原库和原数据卷 | 独立数据卷，由切换前快照初始化 |
| 图片 | /tmp/wcm 原目录 | RustFS 私有桶 wcm-dev，前缀 wcm/images |

旧版代码基线为 `8a261fe`，标签 `single-node-baseline-2026-09-14`；新分支为 `codex/cluster-architecture`。
旧镜像保留为 `wcm-api:single-node-8a261fe-20260914`，旧配置备份位于
`/home/aigc/wcm/backups/single-node-20260914-171236`。

两套程序仍连接同一个 InsightFace，因此备用入口不是人物库的数据快照。
人物写入应只通过正在使用的版本进行，旧程序不参与新集群的写锁协调。
若回退到旧程序，需要先将新版新增图片从对象存储复制回旧图片目录。

## 审核生命周期

`POST /api/v1/review_tasks` 接受 `url`、`sample_interval`、`top_k`、`threshold`，返回 HTTP 202 和持久任务记录。
WebUI 提交一次后订阅任务；页面离开、WebSocket 断开和 API 重启不会取消后台任务。
原 `/analyze_media` 和提交 WebSocket 保留兼容入口，集群模式下也只负责提交、等待和转发。

任务状态为 `queued → processing → completed/partial/failed`；取消经过 `cancelling → cancelled`，排队任务可以直接取消。
MySQL 短事务在全局名额检查后通过 `FOR UPDATE SKIP LOCKED` 领取任务，默认总并发 4。
执行租约默认 60 秒、心跳 5 秒，最多尝试 3 次。Worker 故障后其他 Worker 可重新执行整段视频。
结果、失败、进度和取消完成写入都检查执行令牌及租约期限，过期执行者不能覆盖新执行结果。
本轮不实现窗口级断点续跑；下载缓存和解码临时文件属于 Worker 本地可丢弃数据。

提交时保存业务参数快照，模型参数在任务及其线程中固定使用该快照。
集群总并发和模型请求限额读取实时配置，分别控制 InsightFace、OCR、visual 和 guard。
模型名额和人物写锁由 MySQL 连接持有，不依赖同一机器的文件锁。

## 事件和共享文件

Redis Pub/Sub 只传进度和状态通知，MySQL 是任务状态依据。订阅重连读取快照，服务器每 20 秒校准一次；
提交页另外每 5 秒查询一次，修复通知缺失或网络中断后的显示。
同一次执行忽略倒序 sequence，新执行通过 attempt 区分，允许进度从头开始。

人物元数据中原来的 `/tmp/wcm/...` 保留为逻辑标识，按相对路径映射到对象键，`/images/...` 地址继续有效。
人物展示、图库评分、追加、合并、分类变更和补偿都读取共享后端。证据中的嵌入图片转存为稳定图片引用。
S3 桶保持私有，API 提供兼容图片入口，支持 Range 和 ETag。下载临时文件不进入公共图片目录。

人物写入先保存数据库操作日志和修改前快照，操作记录使用版本条件更新。
中断操作由持有全局人物写锁的实例恢复，再接受下一次写入；照片从共享存储恢复。
人物接口支持 `Idempotency-Key` 重放同一请求；读取返回的 `revision` 可通过 `If-Match` 防止覆盖过期资料。
外部 InsightFace 调用不属于 MySQL 事务，采用补偿恢复；这不等于已经将 InsightFace 改为高可用服务。

## 启动和更新

配置项写在部署目录的 `.env`，文件权限 0600。S3 密钥和数据库密码不要提交到 Git。
必填项包括数据库密码、MySQL root 密码、S3 endpoint/bucket/access key/secret key。
业务参数仍在 WebUI 的参数配置页管理。

```sh
cd /home/aigc/wcm-cluster
sudo docker compose build api webui
sudo docker compose up -d --wait
sudo docker compose ps
sudo docker compose logs --tail 100 worker
```

扩展实例使用 `docker compose up -d --scale api=3 --scale worker=3`。
WebUI 的 Nginx 使用 Docker DNS 动态发现 API，API 和 Worker 不固定容器名、不依赖共享本地图片目录。
同一个任务队列、Redis、S3 和模型限额配置应被所有节点使用。
关闭 Worker 时先停止领取新任务，默认等待 120 秒处理当前任务，超时退出后由租约机制接管。

默认 `compose.yaml` 和 `compose.insightface.yaml` **不设置 CPU 配额**，容器可使用主机可用 CPU；
内存上限、任务与窗口并发限制继续生效。若确需硬限制，显式追加可选文件：

```dotenv
COMPOSE_FILE=compose.yaml:compose.insightface.yaml:compose.cpu-limits.yaml
WCM_API_CPUS=2
WCM_WORKER_CPUS=2
WCM_FACE_SYNC_CPUS=1
WCM_IFS_REPLICA_CPUS=4
```

此示例同时加载副本服务定义，是否启动副本仍由 `face-replication` profile 控制。
CPU 环境变量仅在加载可选文件后使用；只配置环境变量不会开启配额。
不要把上述示例默认写入部署环境。硬配额过低会导致 CPU 推理被 CFS 节流，降低实际并发吞吐。
更新存量部署时应核验容器的 `HostConfig.NanoCpus`：取消配额后的值为 `0`。
变更 CPU 配额无需重启 InsightFace，可按 Compose 服务标签找到当前容器后使用 `docker update --cpus 0`；
同步更新 Compose 文件以保证之后重建仍不设置配额。

## 迁移和验收

迁移命令只复制和核验，不删除旧图片：

```sh
sudo docker compose run --rm --no-deps -v /tmp/wcm:/legacy-images:ro api \
  python -m scripts.migrate_shared_images /legacy-images
sudo docker compose run --rm --no-deps api python -m scripts.migrate_review_evidence
```

图片迁移核验每个对象的大小和 SHA-256；旧人物操作 JSON 导入 MySQL。
重复迁移会校验已有对象，遇到不同内容会停止，不静默覆盖。

`python -m scripts.verify_cluster` 必须使用新建且以 `wcm_verify_` 开头的隔离测试库。
它验证并发领取、统一名额、执行令牌、冻结参数、跨进程模型名额、人物补偿和幂等、跨 API 图片/Range、
Redis 广播、API 退出不影响任务、Worker SIGKILL 后接管以及跨 API 取消。
测试子进程替换模型处理，不对真实 InsightFace 做推理或人物写入。

该部署用于单台 251 上验证多实例行为。MySQL、Redis、RustFS 和 InsightFace 的跨主机高可用不在本轮范围内。

### 2026-09-14 验收记录

- 新版运行 2 个 API、2 个 Worker、独立 WebUI、MySQL 和 Redis，共 7 个健康容器；8000 与旧版 8001 均返回健康状态。
- 图片迁移并逐个校验 15,270 个对象，共 1,277,809,993 字节；22 个原人物操作日志已导入，旧文件保留。
- 24 条历史审核任务保留；检查 19 条已有结果，无需转换其中的嵌入图片。
- 后端测试 643 项通过、7 项外部服务测试跳过；前端测试 84 项通过，生产构建成功。
- 使用隔离 MySQL 库、真实 Redis/S3 和测试模型进程的跨实例验收全部通过，覆盖 API 退出、Worker 强制退出后接管、跨实例取消、旧执行令牌失效和人物补偿。
- 8000 入口实测双 API 分发、提交排队任务后取消、历史图片 SHA-256 和 Range；临时任务已删除。浏览器图库图片全部加载，任务页显示历史记录和新增排队状态。
- 前端发布资源为 `index-BGTI04W4.js` 和 `index--6t5NvB5.css`。InsightFace 原容器保持运行，未重启或迁移 SQLite。

部署日志和数据库快照保存在 `/home/aigc/wcm-cluster/backups/`；本次实例故障测试使用替代模型，未对真实模型做吞吐量或识别效果验收。

参考：[MySQL 锁定读](https://dev.mysql.com/doc/refman/8.4/en/innodb-locking-reads.html)、
[Redis Pub/Sub 投递语义](https://redis.io/docs/latest/develop/pubsub/)、
[Nginx 动态上游](https://nginx.org/en/docs/http/ngx_http_upstream_module.html)。
