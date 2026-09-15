# 项目交付约定

- 用户已明确要求：每次完成功能变更或修复，都要一并更新 251 服务器，不能只停留在本地。同步部署属于常规交付范围，无需每次重新询问。
- 新架构部署目标：`10.252.25.251`，项目目录 `/home/aigc/wcm-cluster`，服务地址 `http://10.252.25.251:8000`，Compose 项目 `wcm-cluster`，服务 `webui`、`api`、`worker`。API 和 Worker 支持多个副本，不能假设固定容器名。
- 旧版备用部署保留在 `/home/aigc/wcm`，入口为 `http://10.252.25.251:8001`，容器 `wcm-facerec-api`。未经用户要求不要覆盖旧版目录、镜像或数据库；InsightFace / SQLite 本轮保持现状。
- 本地验证通过后，检查服务器独立修改和正在处理的审核任务，保留可回滚备份，构建成功再切换 API/WebUI 容器；不要覆盖独立修改或中断正在进行的审核任务。
- 更新后验证服务健康状态和前端资源版本；交付时明确报告 251 是否已同步，遇到阻碍应说明具体原因。
- InsightFace 副本使用 `compose.insightface.yaml` 和 `face-replication` profile，服务为 `insightface-a`、`insightface-b`、`face-sync`；遵循 `docs/insightface-replication.md` 的隔离、备份和恢复步骤。原主节点及 SQLite 格式不修改。不确定写入需要先终止旧执行者和在途请求，再显式恢复，不能仅清除隔离状态。
- S3 模式人物 metadata 使用完整对象 Key 字段 `image_key/image_keys`；统一通过 `image_store` 解析，兼容读取旧 `file_path/image_paths`。字段迁移及回滚遵循 `docs/image-object-keys.md`，通过现有 InsightFace API 和 WCM 可靠日志执行；切回 8001 旧代码前先回滚元数据并核验图片可访问。
