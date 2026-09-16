# 用户、登录与操作权限

## 使用流程

- 用户自行注册，用户名为 3–64 位英文字母、数字、`_ . @ -`，不区分大小写；显示名称支持中文。
- 首个**成功注册**的用户成为超级管理员，数据库事务保证并发注册只产生一个首任超级管理员。之后注册均为普通用户；注册请求不能指定角色。
- 密码为 12–128 个字符。登录后在「账户安全」绑定 Passkey、验证器 2FA，或修改密码。
- 超级管理员在「用户与权限」授予/撤销普通管理员角色、启用/停用账户、调整各角色的操作权限。不能通过页面或 API 降级/停用超级管理员。
- 账户停用或角色改变会退出该用户所有登录；修改密码、启用/关闭 2FA 会退出其他设备。权限配置在下一次请求/消息时生效，前端在重新聚焦及每分钟同步。

## 默认权限

| 操作 | 普通用户 | 普通管理员 | 超级管理员 |
|---|---|---|---|
| 查看人物、检索人脸 | ✓ | ✓ | ✓ |
| 新增、修改、合并、删除人物与照片 | — | ✓ | ✓ |
| 提交审核、查看和下载审核结果 | ✓ | ✓ | ✓ |
| 取消、删除审核任务 | — | ✓ | ✓ |
| 参数配置、系统管理 | — | ✓ | ✓ |
| 用户管理、角色授予、角色权限配置 | — | — | ✓ |

人物和审核记录是团队共享资料，当前按操作权限控制，未引入个人私有数据或任务所有权。
超级管理员可以调整普通用户和普通管理员的业务权限；参数配置、系统管理不能授予普通用户，
用户/角色管理不能通过权限配置授予普通管理员。授予写入或提交权限时，自动补齐对应查看权限。

后端对所有业务 HTTP、图片、OpenAPI 和 WebSocket 入口执行校验。
`/ws/analyze_media` 在握手时校验查看权限，订阅消息使用查看权限，新审核消息额外校验提交权限。
界面隐藏无权限的菜单和操作；手工构造请求也会被后端拒绝。

## 2FA 与 Passkey

验证器采用 TOTP（30 秒、6 位、允许前后一个时间步的时钟偏差）。必须验证成功后才启用。
同一个验证码只能使用一次，刚用于绑定或重新验证的验证码不能再次使用，请等待下一个验证码。
绑定时返回 10 个一次性恢复码，只显示一次，可下载离线保存；数据库只存恢复码 SHA-256。
「用户与权限」和「账户安全」的每次修改均须单独完成二次身份验证；刚登录和刚完成其他验证也不能跳过。
覆盖修改密码、启用/关闭 2FA、重新生成恢复码、绑定/移除 Passkey、修改用户角色、启停用户及保存角色权限。
账户安全使用单列卡片，顺序为 2FA、Passkey、登录密码；修改密码在对话框中输入并确认。

弹窗优先使用当前账户已绑定且浏览器可用的 **Passkey → 2FA → 密码**。
已绑定 2FA 时输入尚未使用的验证码或一次性恢复码；未绑定 2FA 时使用密码。
Passkey 用户可主动切换至下一个可用方式；验证失败不会自动改用较弱方式。
取消验证不会发送修改请求，失败请求也不会自动重试。

验证成功签发一次性操作凭证，绑定当前用户、会话、HTTP 方法和目标资源，在数据库写事务内原子消费。
该凭证仅授权一次操作，不能复用、跨操作、跨用户或跨会话使用；120 秒过期只限制未使用凭证的寿命，不提供免验证窗口。
并发请求争用同一凭证时仅允许一个成功。登录及重新验证不再产生任何可复用的“最近已验证”状态。
验证本身不轮换登录会话，避免破坏正在进行的绑定流程；密码、2FA 等变更仍按原规则撤销其他登录。

绑定 2FA / Passkey 时，开始绑定前消费一次操作凭证，之后仅能用该用户会话中对应的一次性 challenge 完成这次绑定。
确认步骤仍验证新验证码或 WebAuthn 证明，不能用绑定 challenge 授权其他修改。
升级前未记录本次验证授权的旧绑定 challenge 会被拒绝，需重新开始绑定。

Passkey 使用 WebAuthn，要求设备持有证明及用户验证（指纹、面容或设备 PIN）。
使用 Passkey 登录时无需再输入 TOTP；密码登录在启用 2FA 后必须经过第二步验证。
服务端校验 challenge、Origin、RP ID、用户句柄、用户验证标志、签名及签名计数。
登录、注册及验证器绑定 challenge 5 分钟过期且只能使用一次，绑定 challenge 与当前会话关联。
Passkey 重新验证使用独立用途的 challenge，绑定当前用户和当前会话，不能用登录 challenge 或他人的 Passkey 代替。
重新验证只返回当前操作的一次性凭证，不返回免验证会话。

**251 当前只有 `http://10.252.25.251:8000`：密码和 2FA 可用，浏览器不允许在此入口使用 Passkey。**
不要通过关闭浏览器安全检查解决此限制。提供 HTTPS 域名后配置如下：

```dotenv
WCM_AUTH_ORIGINS=["https://wcm.example.com"]
WCM_AUTH_RP_ID=wcm.example.com
WCM_AUTH_COOKIE_SECURE=true
```

RP ID 只写域名，不含协议、端口和路径。HTTPS 在可信反向代理终止，代理转发至 8000；
浏览器访问的完整 Origin 必须显式列入白名单。改变 RP ID 后旧域名的 Passkey 不能直接迁移使用。
本地开发默认允许 `http://localhost:5173`、`http://localhost:8000`，RP ID 为 `localhost`。

## 持久化和配置

集群复用 WCM 的 MySQL 连接配置，新增 `wcm_users`、`wcm_sessions`、`wcm_passkeys`、
`wcm_auth_challenges`、`wcm_role_permissions`、`wcm_auth_limits`、`wcm_auth_mutex`、`wcm_auth_audit` 表。
所有 API 副本共享会话、限流、验证码消费和权限。启动建表使用 MySQL 命名锁，认证写事务使用固定行锁。
无需修改 InsightFace/SQLite、人物 metadata、审核任务表或原有 Worker。

必填集群配置 `WCM_AUTH_SECRET_KEY` 是 Fernet 密钥，用于加密验证器密钥，**所有 API 副本必须一致**。
部署时在服务器生成并保存到权限为 0600 的 `.env`；同数据库备份一起安全保存，丢失后无法解密原有 2FA。
不能每次发布重新生成，不能将真实密钥提交到 Git。生成命令：

```sh
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

本地未启用 MySQL 时使用 `data/auth.sqlite3` 和权限为 0600 的 `data/auth.key`（已忽略 Git）。
集群模式禁止 SQLite；`WCM_AUTH_DATABASE_URL` 可指定独立数据库，留空时复用 MySQL。
会话默认 12 小时绝对有效期，可用 `WCM_AUTH_SESSION_HOURS` 设置为 1–168 小时。
Cookie 为 HttpOnly / SameSite=Lax；HTTPS 部署须启用 Secure。密码采用 Argon2id。
会话令牌仅存哈希；登录按用户与来源限流；写请求校验 Origin 和会话 CSRF 令牌。
认证审计表记录注册、登录、安全设置与角色变更，不记录密码、验证码、恢复码或私钥。

## API 调用

所有路径前缀 `/api/v1/auth`：

- `POST /register`：`username, display_name, password`；`POST /login`：`username, password`。
- 登录返回 `user, csrf_token` 及 Cookie，或 `mfa_required, challenge_id`。
- `POST /login/2fa`：`challenge_id, code`，完成密码登录的第二步。
- `GET /me`：当前用户与 CSRF；`POST /logout`：退出当前会话。
- `POST /reauthenticate`：`operation, method: password, password, code?` 或 `operation, method: totp, code`，返回 `verification_token`。
- `operation` 为实际目标，例如 `POST /api/v1/auth/password` 或 `PUT /api/v1/auth/users/{id}/role`；修改请求通过 `X-WCM-Verification` 头提交凭证。
- `POST /password`：`new_password`；`/2fa/disable`、`/2fa/recovery-codes` 传空对象。均需独立操作凭证，内联密码/验证码不会替代凭证。
- `GET /security`；`POST /2fa/setup`、`/2fa/confirm`、`/2fa/disable`、`/2fa/recovery-codes`。
- `POST /passkeys/register/options`、`/passkeys/register/verify`、`/passkeys/login/options`、`/passkeys/login/verify`。
- `POST /passkeys/reauthenticate/options`：传 `operation`；`/passkeys/reauthenticate/verify`：传 `challenge_id, credential`，返回绑定该操作的 `verification_token`。
- `DELETE /passkeys/{id}`；`GET /users?page=1&query=...`；`PUT /users/{id}/role`、`/users/{id}/active`。
- `GET /roles`；`PUT /roles/{role}`，传 `permissions` 字符串数组。

脚本调用需自行登录并保存 Cookie，公开认证 POST 带 `X-WCM-Client: web`；后续写请求带
`X-CSRF-Token: <登录响应的 csrf_token>`。浏览器 WebSocket 自动携带 Cookie，Origin 必须受信任。
健康检查 `/api/v1/health` 保持公开，不需要登录。认证错误和受保护数据响应均禁止缓存。
敏感操作缺少凭证时返回 `403` 和 `X-WCM-Reauth: required`；前端在发送每次修改请求前主动验证身份，
不依赖拒绝后的重试，也不缓存验证成功状态。已消费或过期凭证返回错误，需重新发起操作并验证。


## 发布与回滚

1. 比对服务器源文件与上次部署基线，发现独立修改时保留并人工合并；检查在途审核。
2. 保存当前源文件、`.env`、MySQL 一致性快照以及 API/WebUI 旧镜像标签。
3. 在独立发布目录构建 API/WebUI 镜像；用隔离数据库验证首次注册并发和跨实例会话，不能占用生产首个注册账户。
4. 为生产 `.env` 首次生成认证密钥；仅切换 API 和 WebUI 容器，并确认所有 API 副本健康。
   原 Worker、face-sync、InsightFace 副本和 8001 旧版入口不重启。
5. 核验健康、资源版本、未登录请求被拒绝、生产账户数仍为零（若用户尚未注册）。

回滚使用保存的旧 API/WebUI 镜像与源文件；保留新增认证表和密钥，不删除已注册用户。
旧代码没有认证功能，回滚意味着旧入口重新开放原有访问方式，需要在受控网络内进行。
本功能不修改图片对象键，也不需要回滚人物 metadata；若另外切回 8001，仍遵循既有图片迁移文档。

## 验证

`tests/test_auth.py` 使用真实 SQLite、Argon2、TOTP 和 ES256 WebAuthn 签名验证器，覆盖首次注册并发、
权限与角色升级限制、会话/挑战撤销、CSRF、限流、恢复码和 TOTP 防重放、图片和 WebSocket 保护。
原业务测试使用隔离的已登录操作员身份；认证专项使用真实中间件，不关闭认证。
`webui/tests/passkeys.test.js` 验证浏览器二进制字段序列化。

参考：[WebAuthn Python 实现](https://github.com/duo-labs/py_webauthn)、
[OWASP 认证建议](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)、
[OWASP 会话管理](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)。

### 2026-09-15 验证记录

- 后端完整回归 722 项通过，7 项外部服务测试跳过；前端 100 项通过，生产构建成功。
- 认证专项 27 项通过；使用真实 ES256 签名校验 Passkey，并覆盖跨用户、错误 Origin/RP ID、用户验证标志、句柄与重放拒绝。
- 251 隔离 MySQL 中启动 2 个独立 API 进程：4 个并发注册仅产生 1 个超级管理员，跨实例会话、角色撤销、权限更新、2FA challenge、恢复码防重放及退出均通过。
- 浏览器核验登录、退出、账户安全、用户列表和普通用户菜单；实体 Passkey 设备操作待 HTTPS 域名配置后验收。

### 身份验证弹窗更新

- 重新验证按可用 Passkey、2FA、密码的优先级展示；该版本的会话时间窗口已由下方逐次验证机制替代。
- 后端完整回归 738 项通过、7 项外部服务测试跳过；认证专项 43 项通过；前端 106 项通过。
- 浏览器核验密码和 2FA 验证后继续操作、错误输入、取消、Passkey 优先及备用方式切换、手机宽度布局。

### 2026-09-16 逐次验证更新

- 取消 5 分钟免验证逻辑，两个页面的每次修改均需一次独立验证；一次性凭证在数据库事务中消费。
- 账户安全按 2FA、Passkey、登录密码单列铺满；修改密码使用独立对话框。
- 验证覆盖凭证重放、并发争用、目标操作/用户/会话隔离、验证取消以及连续两次修改必须分别验证。
- 本地后端回归 748 项通过、7 项外部服务测试跳过；前端 106 项通过，生产构建成功。
- 浏览器核验单列宽度、密码弹窗与焦点恢复；连续修改和跨页面保存均在提交前重新弹出验证。
