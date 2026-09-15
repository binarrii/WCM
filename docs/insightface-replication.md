# InsightFace 可靠副本同步（第一版）

保留原 InsightFace 镜像、接口及 SQLite 格式。WCM 负责可靠写入日志、状态同步、读流量准入和恢复；
原主节点仍为 `10.252.25.251:18097`，新增两个独立数据目录的副本，不共享正在运行的 SQLite 文件。

## 写入与读取约定

写入成功表示：主节点业务操作完成，且 MySQL 在同一事务中提交人物操作结果、同步日志和人物目标状态。
它不表示所有副本已完成。HTTP 超时后应携带原 `Idempotency-Key` 重试；上传表单指纹忽略随机 multipart boundary，
同一键对应不同人物、文件内容或参数会被拒绝。MySQL 提交回执丢失时先查询持久结果，不直接回滚已提交操作。

每条同步记录保存受影响人物的完整目标状态，包括删除标记、人物 ID、完整元数据、照片路径和 SHA-256。
API 写入前保存可补偿快照，并记录每次在途修改。副本只消费已提交记录。
图片先保存到 RustFS，写入日志只引用共享对象；同步实际读回图片并校验 SHA-256 后才允许删除或重建人物。

管理页的列表、统计和同名查询固定读主节点，并与人物写锁协调，保证修改成功后能看到最新资料。
搜索及检测在 WCM 适配层选择达到当前已提交版本、且健康心跳新鲜的副本。
同一次搜索的检测、检索和 `matched_face_id` 后续查询固定使用同一个实例。
因此无需强行统一 InsightFace 内部生成的 `face_id`；人物 ID 仍保持一致。
重建后内部人脸 ID、创建时间以及重新推理得到的浮点特征可能不同；保证的是人物、照片集和元数据的一致，
不保证 SQLite 字节或浮点检索分数完全相同。
Nginx 继续作为 WCM API 入口，不对 InsightFace 的每个子请求独立随机分流。

副本更新前先关闭新读准入，再等待读取租约结束。租约续期失败会取消整个读取，SDK 在下一次 HTTP 调用前再次检查。
副本只有在整个批次完成并保存检查点后才恢复准入，且检查点必须等于当前提交序号。
没有追平副本时回退到受写锁保护的主节点；主节点有未恢复操作时拒绝读取部分状态。
网络在查询中途失败时该查询可能报错，重试应从整个查询开始，不能只把后续 face ID 查询切换到别的实例。

## 性能策略

- 人物姓名、备注等元数据变化通过 PATCH 同步；照片清单未改变时不重新跑人脸注册。
- 照片变化只重建受影响人物，使用批量注册接口，一次校验完整照片集，避免逐照片重试造成重复。
- 每个副本顺序处理日志，默认每批最多 50 条；同一批多次修改同一个人物只应用最后状态。
- 两个副本独立追赶，一个副本的故障不会阻塞另一个，也不阻塞主节点完成已记录的写入。
- 新副本先从原生一致性备份初始化，然后核验完整人物清单，避免首次启动重新推理整个库。
- 沿用 WCM 的集群模型并发限制。`insightface_concurrency` 是全部实例的总准入上限，增加副本不会自动提高这个值。

## 251 部署

部署目录为 `/home/aigc/wcm-cluster`。启用时 `.env` 中设置：

```dotenv
WCM_INSIGHTFACE_REPLICATION_ENABLED=true
WCM_INSIGHTFACE_REPLICAS={"a":"http://insightface-a:8080","b":"http://insightface-b:8080"}
WCM_IFS_REPLICA_IMAGE=wcm-insightface:primary-20260914
COMPOSE_FILE=compose.yaml:compose.insightface.yaml
COMPOSE_PROFILES=face-replication
```

原服务镜像被固定成本地副本标签，两个新增实例不向宿主机发布端口。
`insightface-replicas/a/data` 与 `insightface-replicas/b/data` 完全独立；模型目录只读共享。
复制的运行环境位于 `.env.insightface-replicas`（0600，已加入 Git/Docker 忽略规则）。
数据库文件及配置按原镜像 UID/GID 10001 设置权限，原 `/opt/binarii/insightface-server` 不改动。

启动副本、切换已启用同步的新 API 后，在同步 Worker 启动前执行基线核验：

```sh
cd /home/aigc/wcm-cluster
sudo docker compose run --rm --no-deps api python -m scripts.face_replication seed-all
sudo docker compose up -d --wait face-sync
sudo docker compose run --rm --no-deps api python -m scripts.face_replication status
```

`seed-all` 在人物写锁内核对原库和两个副本的全部受管集合。核验失败的副本保持隔离。
已经初始化时，主节点还必须匹配 MySQL 中的已提交清单，防止把绕过 WCM 的写入静默纳入基线。
未建立基线时，新版本拒绝人物写入；审核任务数据库和旧版 8001 入口保留。

历史记录若缺少完整原照片，只允许从原生一致性备份建立副本，并额外核验完整人脸记录指纹。
这类记录在基线清单中标记 `rebuildable=false`，保持可读，禁止普通业务修改，不能靠剩余照片重建。
图片对象 Key 迁移可在核对原生人脸指纹后仅修改元数据，详见 [图片对象 Key 与回滚](image-object-keys.md)。
必须保留包含它们的原生备份；后续补齐原照片需要单独核验迁移。`seed-all` 输出这类记录的数量。

查看 `GET /api/v1/insightface/replication` 可获得提交序号、每个副本的检查点、落后序号差、心跳、重试次数、
下次重试时间、隔离原因以及主节点待恢复状态。MySQL 的 `face_sync_changes` 和 `person_operations` 保存审计与恢复记录。

## 故障恢复

| 故障 | 系统行为 | 恢复方式 |
|---|---|---|
| 副本在写入前已不可达 | 退出读池，指数退避，最多间隔 300 秒 | 服务恢复后自动追赶 |
| 图片读取或校验失败 | 不开始破坏性修改，不推进检查点 | 修复对象存储后重试 |
| 写入超时、回执丢失、注册部分失败 | 副本隔离，不自动重放未知写入 | 隔离旧执行者和在途请求后恢复 |
| 同步进程在批次中被杀死 | 新执行者发现遗留状态后隔离副本 | 停止旧进程、重启对应副本，再恢复 |
| 主节点写入或补偿结果不确定 | 保留操作快照，阻止后续人物写入 | 隔离主节点在途请求后补偿恢复 |
| 副本数据卷丢失 | 不允许空库进入读池 | 隔离副本，从主节点一致性备份重建，再 seed |
| 主节点恢复到旧备份 | 持续阻止主节点读写，直到完整核验 | 按备份记录的序号重放已提交日志 |

**`--fenced` 是操作者对“旧执行者和在途请求已经停止”的确认，工具不会凭这个参数自动隔离网络。**
未修改的 InsightFace 不能执行 WCM 的 fencing token。仅凭 MySQL 锁超时就重试旧写入无法保证安全，
所以涉及不确定写入时选择停止服务能力，而不是冒险继续自动写入。普通断网且尚未尝试写入的情况可以自动恢复。

副本 A 不确定写入的操作顺序：

```sh
# 先停止所有连接这个副本的同步 Worker；多主机时必须在全部相关主机上停止。
sudo docker compose stop face-sync
sudo docker compose run --rm --no-deps api python -m scripts.face_replication quarantine --node a
# 重启该副本，确保旧请求已经终止。原主节点不需要重启。
sudo docker compose restart insightface-a
# 等待 insightface-a 健康，并确认不存在其他绕过 WCM 的写入进程后：
sudo docker compose run --rm --no-deps api python -m scripts.face_replication recover --node a --fenced
sudo docker compose up -d --wait face-sync
```

恢复从已确认检查点继续；重放完整人物状态，不重放“再加一张照片”的不确定命令。
若副本数据卷已丢失，必须恢复一致性备份并执行 `seed --node a`，不能只执行 `recover`。
`seed` 要在所有同步 Worker 停止、旧副本写入已隔离后使用。不要让两个容器挂载同一 SQLite 数据目录。

主节点操作补偿使用 `recover-primary --fenced`。执行前停止人物写入源，等待正在审核的任务结束，
停止旧写入进程并终止主节点在途请求，完成隔离后再执行；工具不会自动抢占或提升副本。

主节点磁盘故障需先离线恢复一份完整的一致性备份，记录该备份的已提交序号，再执行：

```sh
sudo docker compose run --rm --no-deps api python -m scripts.face_replication replay-primary --after-sequence 0 --fenced
```

上面的 `0` 仅适用于本次同步功能启用前的基线备份，其他备份必须使用各自记录的序号。
工具通过原 HTTP API 重放备份后的已提交目标状态，并核对全量人物清单。失败或中断时，持久恢复标志保持开启，
不能通过重启 API 绕过。成功后才解除主节点读写限制。整个过程不编辑 SQLite 表或修改 InsightFace 源码。

## 约束与验收

一致性保证覆盖通过新版 WCM 统一入口的写入。旧版 8001、InsightFace 自带管理页和独立脚本不能同时写同一人物库；
它们不会生成本同步日志。受管主节点地址和集合配置改变时系统拒绝继续，需要重新规划并核验基线。
本轮没有自动主节点切换，也没有提供 MySQL、RustFS 自身的跨主机高可用。持续可靠恢复需要保留 MySQL 日志、
RustFS 原图和带序号的 InsightFace 一致性备份；不要单独清理同步日志或原图片。

`scripts.verify_face_replication` 使用以 `wcm_verify_` 开头的隔离 MySQL 库和两个新副本中的临时集合，
拒绝连接原主节点作为测试写入目标。验收覆盖真实注册和搜索、HTTP 表单幂等重试、元数据免重新注册、
批次合并、断网追赶、读取租约排空、写响应丢失、进程 SIGKILL、MySQL 提交回执丢失、删除同步及主节点备份重放。
测试集合和测试图片在退出时清理，不修改原人物库。

### 2026-09-14 验收记录

- 本地后端：658 项通过，7 项有条件跳过；Ruff 与 diff 格式检查通过，WebUI 源码未变。
- 251 的隔离真实 HTTP 验收全部通过，包括实际 SIGKILL、写入完成但响应丢失、MySQL COMMIT 回执丢失、
  错误备份序号阻止恢复，以及正确序号重放后全量核对。
- 同机单次样本：主节点元数据修改约 209 ms；三次元数据修改合并同步约 253 ms。
  注册验收段包含 HTTP 注册和同键重试两次调用，共约 7.48 秒；这是验收计时，不是吞吐压测结论。
- 基线包含跨受管集合的 9,803 条人物记录。其中 1 条历史聚合记录有 6 张人脸、仅 1 张原照片路径，
  通过原生备份及人脸指纹保留，禁止修改；未对原人物库进行修补。
- 上线后两个副本均为 `ready`、序号差 0；真实照片检索分别命中 A、B，连同人脸框回查均成功。
  API、审核 Worker 各两个实例，两个 InsightFace 副本及同步 Worker 均健康，8000/8001 入口正常。
- 251 日志位于 `backups/face-replication-verification-d.log`，基线核验日志为 `backups/face-replication-seed.log`。
  启用前备份在 `backups/before-face-replication-20260914-183853/`，包含序号 0 的原生备份和 MySQL 转储。
