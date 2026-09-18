# Qwen 旗帜与徽标 bbox 独立测试（2026-09-18）

`WasuAI/Qwen3.8-27B-Abliterated` **能够识别旗帜/徽标并输出边界框**。
适合继续做真实画面的候选框实验；本轮发现坐标制混用、普通图形误报和 Markdown 包裹，尚不能作为可靠检测器直接接入审核结论。

分支：`codex/object-detection-evaluation`，起点 `b31bcf1`。
调用部署中的模型网关，响应的模型名称为 `qwen3.8-27b-abliterated`。
没有核验网关后的权重哈希，因此不把它视作官方原版模型的效果评测。

- [可视化报告](report.html)：绿色真实框，粉色模型框；每张图保留完整文本响应。
- [机器可读统计](summary.json)、[样本与标注](manifest.json)。
- [首轮归一化响应](responses-normalized.jsonl)、[简短提示词响应](responses-normalized-compact.jsonl)、[像素坐标响应](responses-pixels.jsonl)。
- [可重跑脚本](../../../scripts/eval_flag_logo_bbox.py)。

## 官方资料说明了什么

查阅日期：2026-09-18。区分具体型号模型卡、官方系列教程和仓库社区答复，避免把其他型号的成绩移用于本模型。

1. 用户提供的 [Qwen/Qwen3.8-27B 模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) 明确说明这是原生图像/视频模型，提供图像输入和关闭思考模式的 API 示例；本次查阅的正文没有单列 bbox/grounding 规范或旗帜/徽标专项成绩。
2. [Qwen3-VL 技术报告 §3.2.4](https://arxiv.org/html/2511.21631v1#S3.SS2.SSS4) 明确介绍框定位、点定位与计数训练，坐标归一化到 `[0,1000]`；§5.5 评测 RefCOCO/+/g、ODinW-13 等定位任务。这说明该系列有受训的定位能力，但不是 Qwen3.8-27B 或 Wasu 改版的专项成绩。
3. [官方 2D grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb) 提供多目标定位、JSON、`bbox_2d` 和画框代码。实际代码按 `[x1,y1,x2,y2]` 读取，x 乘宽度、y 乘高度，均除以 1000。教程一个绘图函数的 docstring 写成了 y/x 顺序，**应以其转换代码及实际提示示例为准**。
4. [QwenLM 官方仓库中的 Q&A #56](https://github.com/QwenLM/Qwen3.8/discussions/56) 的已采纳答复说明 Qwen3.5 沿用 `0–1000` 坐标并指向上述教程。问题针对的是 Qwen3.5，仓库名称现在为 Qwen3.8；不能据此宣称有针对 3.8-27B 的官方保证。
5. [QwenLM 的 grounding 工具实现](https://github.com/QwenLM/Qwen-MM-Plugins/blob/main/src/capabilities/api/qwen_mm_plugins_api/vl/grounding.py) 同样通过图像+提示词获取 JSON 数组，使用 `label` 和 `bbox_2d`，按 `0–1000` 转换为原图像素。无需额外检测头才能请求文本 bbox；此文件本身不证明指定 Wasu 模型的效果。

## 测试方法与边界

程序绘制 11 张 PNG：9 张正样本，共 27 个目标，以及空白、普通几何形状两个负样本。
包括中国、日本、法国国旗及 YouTube、奥运五环、奔驰标志的**简化图案**，覆盖单目标、多目标、重复目标、贴边、横竖图及小目标。
最小目标为 48×27 像素的 YouTube 图案，所在图片为 1280×720。
原始图、尺寸、SHA-256 和精确绘图位置都在 manifest 中；模型只收到图片和通用提示，未收到目标名称清单或真实框。

所有请求单独执行，没有聊天历史，串行调用，`temperature=0`、`max_tokens=2048`、HTTP 超时 60 秒，不自动重试；不额外设置服务端思考开关。
复用现有 visual 并发名额，未调用 OCR、Guard 或人脸服务，也未创建审核任务。
一共 26 次模型调用，全部 HTTP 200、`finish_reason=stop`。

评分用 `flag/logo` 大类一致且 IoU≥0.5 的最大数量一对一匹配；具体名称另外检查，不以同义词字符串相等评分。
只剥离 Markdown 围栏，不自动纠正坐标、不猜单位、不裁剪越界框。
任意一个框越界会使整条响应判为无效，其目标全部记为未匹配；因此像素组统计包含这种协议失败惩罚。
平均 IoU 是每个真实目标与同大类预测框的最佳 IoU 平均值，缺少候选时为 0；不是 mAP。

## 结果

| 提示方案 | 请求数 | 定位成功 / 真实目标 | Precision / Recall | 平均最佳 IoU | 响应中位数 |
|---|---:|---:|---:|---:|---:|
| 归一化，提示中写明原图像素尺寸 | 11 | 24 / 27 | 88.9% / 88.9% | 0.888 | 2.064 s |
| 归一化，简短提示且不写像素尺寸 | 11 | 27 / 27 | 90.0% / 100% | 0.973 | 2.006 s |
| 明确要求原图像素坐标，4 张对照 | 4 | 1 / 12 | 20.0% / 8.3% | 0.113 | 2.285 s |

两次归一化测试使用同一套图片；像素组只测 `single_flag/mixed_wide/small_objects/portrait`，样本不同，不能将各组汇总视为严格统计比较。
简短提示词是在首轮发现单位混淆后追加的探索性对照，不是独立留出集；没有对随机性或跨次稳定性做统计。

### 失败与协议问题

- 首轮贴边图（900×610）返回的法国旗 `[0,31,157,131]`、YouTube `[643,0,820,102]`、奔驰 `[793,500,900,607]` 接近**原图像素**。按要求的归一化坐标解释，3 个框全部失败；这支持发生单位混淆的判断，但未检查模型内部处理。
- 简短提示词修复了本次贴边图的定位；其法国旗返回 `[0,49,167,215]`，奔驰返回 `[878,818,995,995]`，符合归一化位置。但同一提示词把负样本里的普通紫色折线称为 Google 标志，把圆和三角称为 unknown logo，产生 **3 个假阳性**。首轮负样本则返回空数组。
- 像素坐标组仅单旗图片正确。混合图仍返回接近归一化的坐标；小目标图出现 x 近似归一化、y 近似像素的混合；竖图仍采用归一化的 x。因此不宜让模型直接承担像素换算。
- 首轮只有 6/11 条响应是裸 JSON，简短提示只有 1/11 条；其余带 Markdown 围栏。即使提示要求仅 JSON，也需要解析和校验。
- “所有数值都在 0–1000 内”并不能证明坐标制正确，首轮贴边失败就是反例。

## 推荐继续实验的输出约定

使用 `bbox_2d=[xmin,ymin,xmax,ymax]`，范围固定 `0–1000`。
由应用统一换算到**实际发送图片**的尺寸，避免同时在提示中混用像素和归一化单位：

```python
x1, y1, x2, y2 = bbox_2d
pixel_box = [x1 * width / 1000, y1 * height / 1000,
             x2 * width / 1000, y2 * height / 1000]
```

真实响应示例（首轮 single_flag，输入 900×600，已剥离 Markdown 围栏）：

```json
[{"bbox_2d":[197,218,598,617],"label":"China","category":"flag"}]
```

对应像素框约 `[177,131,538,370]`；真实框为 `[177,131,537,371]`。
若后续视频抽帧缩放、裁剪或补边，显示到原视频时还需要相应的反向坐标变换。

目前只建议把这些框用作候选定位与可视化。负样本证明“是否真的存在徽标”还需验证；下一步应使用人工标注的实际截图，覆盖飘动旗面、衣服胸标、台标、水印、遮挡、模糊和未知图案。
本轮未测试真实照片、视频、政治/组织身份判断或违规判断，不能外推真实业务准确率。

## 重跑与 251 同步

本地查看与重新汇总不调用模型：

```sh
uv run python -m scripts.eval_flag_logo_bbox summarize docs/model-evaluations/2026-09-18-flag-logo-bbox
```

在具备现有网关配置的环境运行（使用新文件保存响应，避免覆盖历史结果）：

```sh
uv run python -m scripts.eval_flag_logo_bbox run docs/model-evaluations/2026-09-18-flag-logo-bbox/manifest.json --coordinates normalized --prompt-style compact
```

容器运行外置脚本时保留应用搜索路径：`PYTHONPATH=/app:/app/src`。
本次运行使用 API 第 1 个 Compose 副本的独立 `/tmp/wcm-qwen-bbox-20260918` 目录；前两次启动因模块搜索路径缺失而在调用模型前退出，随后修正路径，无模型请求重试。

脚本、样本、原始结果、标注图和报告作为独立实验归档同步到 251：
`/home/aigc/wcm-cluster/evaluations/2026-09-18-flag-logo-bbox/`。
没有替换业务代码、构建部署镜像、重启容器或修改旧版 8001；本轮交付为独立测试材料，不是生产功能变更。
