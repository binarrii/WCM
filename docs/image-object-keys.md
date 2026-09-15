# 人物图片对象 Key

S3 模式的人物 metadata 使用 `image_key`（主图）和 `image_keys`（完整图库，包含主图）。
值为完整对象 Key，例如 `wcm/images/people/demo.jpg`。Bucket、Endpoint、访问密钥保留在 WCM 配置中。
新注册、追加、合并、删除照片、人物分类调整统一写入新字段；本地文件模式保留旧格式。

旧 `file_path/image_paths` 仍可读取；同时存在两套字段时，新字段优先。新写入清除旧字段。
人物、图片对象和已登记的人脸不因字段迁移而移动或重建。对象 Key 限定在配置的图片前缀内，
禁止目录穿越、隐藏目录及其他 Bucket 前缀。中文及 URL 保留字符会正确编码。

## 图片访问

`image_key=wcm/images/people/demo.jpg` 生成 `/images/people/demo.jpg`。
WebUI Nginx 转发给 WCM API，API 使用配置中的 Bucket 和完整 Key 从 RustFS 流式读取。
现有 `/images/...` 地址继续有效；图片 API 保留 ETag、HEAD、Range 和缓存支持。

## 迁移与恢复

迁移只通过 InsightFace 现有 PATCH API 修改人物 metadata，不修改服务源码、镜像或 SQLite 表结构。
每条记录经过 WCM 的全局人物写锁、操作快照、幂等键和 MySQL 事务同步日志。
修改前检查计划中的 metadata 哈希和对象存在性；只允许改变图片引用格式。
普通人物和缺原图的历史人物均采用仅元数据补偿，补偿不删除、重建人物。
历史缺图记录另外核对原生人脸指纹，业务编辑限制仍保留。

新增同步记录的图片清单使用 `key`，兼容读取旧日志的 `path`。旧检查点会按同一个对象进行比较，
不会仅因路径改名触发重新注册。旧日志不被覆盖；原生备份恢复后的全量核验兼容两种日志格式。

在 `/home/aigc/wcm-cluster` 执行。先保留带提交序号的原生一致性备份和 MySQL 备份，
确认没有待恢复的人物操作，且两个副本均已追平。以下 report 文件名每次必须使用新名称，权限自动设为 0600。

```sh
# 仅生成计划
sudo docker compose run --rm --no-deps -v "$PWD/backups:/reports" api \
  python -m scripts.migrate_image_keys --report /reports/image-keys-preview.json

# 生成完整回滚记录后逐条执行，每 100 条输出一次进度
sudo docker compose run --rm --no-deps -v "$PWD/backups:/reports" api \
  python -m scripts.migrate_image_keys --apply --report /reports/image-keys-applied.json

sudo docker compose run --rm --no-deps api python -m scripts.face_replication status
```

每次运行先核对当前资料，已经迁移的记录跳过；每份计划有独立 ID，防止回滚后再次迁移时误命中旧幂等结果。
发生不确定写入时停止迁移，先按 InsightFace 同步文档终止旧执行者及在途请求，再执行主节点恢复流程。
恢复完成后使用新 report 文件继续，不能靠直接重启反复重放未知写入。

需要回滚图片字段时，保留新版 WCM 运行，让它向所有副本同步旧字段：

```sh
sudo docker compose run --rm --no-deps -v "$PWD/backups:/reports" api \
  python -m scripts.migrate_image_keys --apply \
  --rollback-from /reports/image-keys-applied.json --report /reports/image-keys-rollback.json
```

回滚前核对部署来源和完整元数据；迁移后被编辑或删除的记录会阻止整份计划执行，避免覆盖后续修改。
多次分段迁移需要保留全部 report，并分别回滚。后来新增的 Key 格式人物也需要单独转换后才能交给旧代码。
**8001 旧版不支持新图片字段：回退到旧版前必须先完成字段回滚及副本追平。** 8001 的代码、镜像和数据库保持原样。
如果旧版使用本地图片目录，还需准备迁移后新增图片的本地副本，不能只转换字段。

旧原生备份和迁移日志必须一起保留。恢复备份后，使用对应提交序号重放日志；缺原图人物仍依赖原生备份保存完整人脸。
