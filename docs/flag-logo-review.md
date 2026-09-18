# 对象检测：关注旗帜/徽标与明确裸露

综合审核使用 `face`、`ocr`、`visual`、`flags` 四个同级分支。为兼容已有任务与部署，
对象检测继续使用 `flags` 接口字段及参数前缀；参数配置页、进度和错误信息显示“对象检测”。
模型为 `WasuAI/Qwen3.8-27B-Abliterated`，每次处理一张图片。

命中均为 `needs_review` 待人工复核发现，保留 bbox，不自动判违规，不根据持有标志判断人物立场或身份。

## 关注范围

| target | 关注内容 |
| --- | --- |
| tibet_related | 雪山狮子旗、ICT、Free Tibet 的专用旗帜/图形徽标 |
| east_turkestan | 东突厥斯坦星月旗及可辨认的相关专用标识 |
| taiwan_related | 中华民国青天白日满地红旗、台湾独立运动专用旗帜/图形徽标 |
| hong_kong_independence | 可明确识别的香港独立运动专用旗帜/图形徽标 |
| pride | LGBT 彩虹旗、进步骄傲旗及明确使用这些图案的徽标 |
| extremist_organization | 可辨认具体极端/恐怖组织的专用旗帜/徽标，必须有独特图案依据 |
| listed_organization | 额外名单中的组织专用标识；默认新唐人、新中国联邦 |
| exposed_genitals | 实际裸露、可辨认的外部生殖器官 |
| exposed_female_nipple | 女性乳房部位实际裸露的乳头/乳晕 |
| exposed_anus | 实际裸露、可辨认的肛门 |

正向提示词列出对象、特征、识别依据；反向提示词排除普通国家/地区与组织标志、常见品牌、
自然彩虹和多色装饰、普通文字、宗教通用图案，以及泳装、内衣、胸沟、遮挡部位和仅凭轮廓的猜测。
明确列入关注范围的标识按名单处理。未知组织不由模型根据“反华”等笼统概念自行扩充。
医学或艺术场景中的实际外部裸露仍可作为复核线索；内部器官剖面和解剖文字本身不算裸露。

## 参数与执行

在参数页“对象检测”中调整以下参数，新提交任务会冻结配置快照。旧分组元数据保持原值，
仅调整界面显示，确保兼容仍运行旧代码的 face-sync 进程。

| 参数 | 默认值/含义 |
| --- | --- |
| flags_enabled | true，控制整个对象检测分支 |
| flags_model | WasuAI/Qwen3.8-27B-Abliterated |
| flags_timeout_s | 50，每次请求的秒数期限 |
| flags_max_tokens | 2048，最大输出 token 数 |
| flags_positive_prompt | 关注范围及视觉特征，保留英文 target 标识 |
| flags_negative_prompt | 排除规则与误报抑制 |
| flags_organization_targets | JSON 名称数组，默认 `["新唐人", "新中国联邦"]`；空数组关闭额外名单匹配 |

正反提示词按区段放入聊天请求文本，不使用网关未支持的 `negative_prompt` 参数。
新增名单中 `新唐人` 和 `新中国联邦` 带有专用图形提示；其他名称以可辨认的专用标识匹配。

图片、窗口视频、旧 target 视频的综合审核均使用此分支。沿用既有采样点和相同图像缓存，
与视觉描述共用 `visual_concurrency`。独立 OCR、人脸和视觉接口不额外调用对象检测。
分支失败重试、错误预算、取消清理与覆盖率沿用现有实现；失败为 `incomplete`，不是无目标。

## 输出与保存

模型必须输出 JSON 数组；每个目标含 `category: flag|logo|nudity`、合法 `target`、
具体 `label`、简短可见依据 `evidence` 和 `bbox_2d: [xmin,ymin,xmax,ymax]`。
`listed_organization` 必须带与配置名单完全匹配的 `organization`。
坐标固定为完整图像的 0–1000 归一化值，无目标返回 `[]`。

名称占位词/unknown、未知 target、类别不匹配、缺少依据、名单外组织、非法 bbox 或截断 JSON
均视为无效模型响应，走失败流程，不以空结果掩盖错误。

持久化结果继续使用原有 JSON 字段，不修改审核结果表结构：

```json
{
  "timestamp": "00:00:02.000",
  "source": "flags",
  "category": "裸露部位",
  "object_type": "nudity",
  "object_target": "exposed_female_nipple",
  "object_evidence": "乳头区域实际可见，未被遮挡",
  "name": "女性乳头/乳晕裸露",
  "description": "裸露部位：女性乳头/乳晕裸露",
  "review_status": "needs_review",
  "object_samples": [{
    "time_ms": 2000,
    "pts_seconds": 2.0,
    "duration_seconds": 0.04,
    "frame_index": 50,
    "bbox": {"x": 0.5, "y": 0.1, "w": 0.05, "h": 0.06}
  }]
}
```

旗帜/徽标类别保持“旗帜与徽标”，裸露使用“裸露部位”。各目标保留独立位置、可见依据和关注类别。
裸露名称使用固定解剖学标签。前端支持分类筛选、记录联动、采样帧跳转及 JSON 导入导出。
框色分别为旗帜橙色、徽标青色、裸露紫色；仍仅在暂停并确认与采样帧 PTS 匹配时显示。
对象框显示时以原中心将宽、高各放大至 1.25 倍，超出画面部分截断；人脸框保持原有大小。
这只调整显示范围，保存及 JSON 导出的模型 bbox 不变，多次刷新不会累积放大。
其中，原始 bbox 的宽、高按视频原始像素计算均不超过 64px 时，改用绿色圆环十字标。
阈值在 1.25 倍扩张前判断，与播放器显示大小、全屏及浏览器缩放无关；任一边超过 64px 仍使用框。
结果卡片仅在时间戳旁保留“待复核”状态标签，类别与说明之间不再重复显示“（待复核）”。
十字标以原始 bbox 中心定位，点击区域固定为 32 个显示像素，画面边缘只截断标记，不移动中心。
点击小标记切换信息浮层显示/隐藏，打开时联动右侧记录；再次点击或 Escape 收起，切换采样帧时清理浮层。
人脸不适用此阈值，较大对象和原有四角框/完整框显示模式继续保持原行为。
实现验收见 [小对象圆环十字标验证](verification/2026-09-18-object-crosshair/README.md)。
不会推断两个采样点之间的持续出现时间，`needs_review` 不增加审核未完成比例。

## 验证与限制

本次 [实测记录](model-evaluations/2026-09-18-object-scope/README.md) 包含正反例、已知漏检和部署信息。
原图现仅保留中华民国旗与新唐人台标；日本、德国、美国国旗被排除。
该版本仍可能漏检过小或模糊的标志：100×70 的新中国联邦旗样例未检出，高清样例检出。
少量样例仅证明流程和这些实例的表现，不代表全类别、真实视频或复杂遮挡条件的准确率。
小部位的框中心仍可能偏移，扩大显示框不等于改善模型定位；
见 [74 秒采样帧定位复查与显示调整](model-evaluations/2026-09-18-object-box-padding/README.md)。

初版全旗帜检测的历史记录见 [2026-09-18 初版验收](model-evaluations/2026-09-18-flags-integration/)。
