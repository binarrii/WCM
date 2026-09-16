# 登录页背景接入与 251 部署记录

日期：2026-09-16。分支：`codex/ui-polish-20260916`。

已采用文案：“人物识别、违规审核，一站完成。”；辅助说明：“管理人物库，识别视频人物，检测违规内容，快速定位风险片段。”

四张 SVG 全部保留。默认进入登录页时随机选择，切换登录 / 注册不重新抽取。
右上角选定背景即固定保存至同源浏览器本地存储；“每次随机”会解除固定。
深浅配色局限于登录页，不改变工作台主题或认证流程。

## 验证

- 前端测试 111 项通过，含随机覆盖四图、固定后重开、解除固定、无效值和存储不可用的五项新增测试。
- Vite 生产构建通过，四张 SVG 使用带内容哈希的静态资源地址。
- 浏览器验证四套背景、固定选择后刷新、恢复随机后重开、登录 / 注册切换。
- 桌面 1280 × 720 登录页完整显示；手机 390 × 844 无横向溢出，注册页可自然滚动。
- 251 上验证选择深蓝光学后刷新仍固定，SVG 成功解码，验收后恢复默认随机模式。
- 线上 HTML、JS、CSS、四张 SVG 和两个图标共九项资源均返回 200，SHA-256 与构建产物一致。

## 部署

入口：`http://10.252.25.251:8000`，目录：`/home/aigc/wcm-cluster`。

先核对服务器源码，保留其两处“创建账户”独立文案修改。
完整比较 69 个前端源码、公共资源与构建配置文件，与隔离发布目录一致。
首轮容器构建因 Docker Hub 连接重置而停止，未影响生产服务。
随后使用已验证的本地 Vite 产物，以服务器当前 WebUI 镜像中的 Nginx 为基础离线构建，未更换 Nginx 配置。

- 新镜像：`wcm-cluster-webui:auth-20260916-064736z`。
- 旧镜像：`wcm-cluster-webui:before-auth-20260916-064736z`。
- Compose 现用 WebUI 标签仍为 `wcm-cluster-webui:concurrency-071c5fb`，已指向新镜像；API 的同名版本标签未变动。
- 备份：`/home/aigc/wcm-cluster/backups/auth-backgrounds-20260916-064736Z/`。
- 该目录包含 `AuthPage.vue.before`、隔离发布源文件及 `deployment.json`（镜像 ID、文件哈希、服务实例与健康记录）。
- 部署前后在途审核均为 0。只切换 WebUI；其他集群容器的 ID 和启动时间保持一致。
- WebUI 和全部集群服务均健康。8001 旧部署未变动。

## 回滚

在 251 的 `/home/aigc/wcm-cluster` 下恢复登录页原始源码，并重新标记及启动已保留的 WebUI 镜像：

```sh
cp backups/auth-backgrounds-20260916-064736Z/AuthPage.vue.before webui/src/views/AuthPage.vue
sudo docker image tag wcm-cluster-webui:before-auth-20260916-064736z wcm-cluster-webui:concurrency-071c5fb
sudo docker compose -p wcm-cluster --env-file .env --env-file .env.insightface-replicas -f compose.yaml -f compose.insightface.yaml up -d --no-deps --wait webui
```

本次不修改数据库、人物 metadata、API、Worker 或认证逻辑，不需要数据回滚。

## 背景选择器收紧（同日）

- 下拉框与“背景”文字均为 20px 高，整体透明度 0.7，图标和文字改用次要色，背景透明、边框弱化。
- 移除“本次：…”及固定成功提示；仅本地保存失败时显示异常说明。随机和固定逻辑不变。
- 生产构建、深浅背景视觉检查及线上 DOM 尺寸检查通过。
- 新镜像：`wcm-cluster-webui:picker-20260916-065933z`；前一镜像备份：`wcm-cluster-webui:before-picker-20260916-065933z`。
- 备份：`/home/aigc/wcm-cluster/backups/auth-picker-20260916-065933Z/`，两份源文件位于其中 `before/webui/src/views/`。
- 部署前核验全部前端源码与基线一致、无在途审核；仅切换 WebUI，其余服务实例与启动时间未变。
- 服务健康，九项线上资源与新构建产物逐一校验一致；资源版本为 `index-D02Ugnij.js` / `index-CLrdZcOc.css`。
