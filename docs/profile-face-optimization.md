# 侧脸与困难人脸识别优化

## 处理流程

这套优化不改变主滑窗或场景采样策略。每个原有选中帧仍只进入一次主审核，随后只对困难人脸做一次有总量上限的临帧补采样：

1. 保留 InsightFace Server 返回的五点关键点、检测分、清晰度、亮度和姿态质量。
2. 视频查询使用带 25% 上下文的方形裁剪；关闭优化开关时仍使用原有紧裁剪。
3. 内部候选召回最低可放宽到 0.35，但最终确认不得低于调用方原有相似度阈值。
4. 同一镜头内按位置和时间把人脸组成短轨迹，同一人物至少两个不同帧的证据一致才确认。正脸且相似度至少 0.65、与第二候选差至少 0.05 时保留单帧快速确认。
5. 侧脸、低清晰度、低分候选或候选差距不足时，尝试读取前后 0.2/0.4 秒的临帧，再进行一次聚合；不会递归补帧。

InsightFace Server 0.2.0 对外提供的是 0～1 的姿态质量分，不是校准过的 yaw/pitch/roll 角度。WCM 另保留一个由五点关键点估算的 yaw 代理值用于诊断，但只使用服务端姿态质量分决定是否进入困难样本路径。

## 性能边界

主采样数量不增加。默认补帧上限同时受两个条件约束：

- 每个审核窗口最多 3 帧；
- 整个视频最多 `ceil(主选中帧数 × 0.30)` 帧。

因此长视频的额外人脸调用目标上限约为 30%；短视频为了让优化能够生效，向上取整可能产生 1 次补帧。补帧并发默认为 2，且移除了视频查询中逐个读取图库样本框的额外请求。日志会记录主/复用人脸调用数、补帧数、确认数和疑似数。

## 结果语义

- `recognition_status=confirmed`：满足高置信单帧规则，或至少两个不同时间点的一致证据达到原阈值。
- `recognition_status=probable`：达到原阈值但证据数、姿态或候选差距不足，保留给人工复核。
- 低于原阈值的内部候选不会成为结果，只能触发一次受限补帧。
- `face_samples` 可包含 `pose_score`、`sharpness`、`quality_score`、`estimated_yaw`、`candidate_margin` 和 `auxiliary` 诊断字段。

## 配置与回滚

| 参数键 | 默认值 | 说明 |
| --- | ---: | --- |
| `face_profile_optimization` | `true` | 总开关；设为 `false` 恢复原有视频判断和紧裁剪 |
| `face_crop_padding` | `0.25` | 优化查询裁剪的单侧扩展比例 |
| `face_candidate_similarity` | `0.35` | 仅供聚合/补帧的内部候选下限 |
| `face_high_similarity` | `0.65` | 正脸单帧快速确认下限 |
| `face_min_candidate_margin` | `0.05` | 第一、第二候选最小差距 |
| `face_min_confirming_frames` | `2` | 普通确认所需不同帧数 |
| `face_profile_pose_threshold` | `0.60` | 低于此姿态质量分视为困难样本 |
| `face_low_sharpness_threshold` | `0.15` | 低于此清晰度视为困难样本 |
| `face_track_max_gap_s` | `2.5` | 同一短轨迹允许的最大时间间隔 |
| `face_neighbor_offsets_s` | `[-0.4,-0.2,0.2,0.4]` | 临帧候选偏移，JSON 数组 |
| `face_max_extra_frames_per_window` | `3` | 单窗口补帧上限 |
| `face_max_extra_call_ratio` | `0.30` | 全视频补帧/主选中帧目标上限 |
| `face_neighbor_concurrency` | `2` | 补帧人脸请求并发 |
| `face_gallery_target_samples` | `5` | 图库每人的目标样本数 |
| `jpeg_quality` | `95` | 包括侧脸查询裁剪在内的 JPEG 编码质量 |

紧急回滚只需在参数配置页面把 `face_profile_optimization` 改为 `false`，不需要重建 API 容器、迁移数据库或改写历史结果。

## 图库覆盖审计

快速统计每人的样本数量（只读，不读取人脸图像）：

```bash
uv run python scripts/audit_face_gallery.py --output gallery-summary.json
```

离线检查是否至少有一张侧脸姿态样本；该模式会按人物读取样本元数据，4900 人规模约需 4900 次以上 API 请求：

```bash
uv run python scripts/audit_face_gallery.py \
  --inspect-quality \
  --include-persons \
  --output gallery-remediation.json
```

补录建议优先满足每人至少 5 张清晰、单人、年龄跨度适中的样本，其中包含左侧、右侧和正脸。审计工具只生成缺口清单，实际补录继续使用现有人物库的“添加照片”能力，避免脚本自动写图库。
