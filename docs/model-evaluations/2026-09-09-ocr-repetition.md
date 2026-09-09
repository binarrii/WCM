# PaddleOCR-VL 1.6 重复输出优化

## 范围与方法

- 分支：`codex/paddleocr-repeat-optimization`。
- 使用项目现有 WasuAI/PaddleOCR-VL-1.6 服务，顺序测试 `test_pieces-2.mp4` 的 0、10、74、580 秒四帧，另加合成纯白图；没有重跑整条视频。
- 所有 OCR 对照保持 `max_tokens=300`、`temperature=0`。通过 ffmpeg 提取长边 1080 的画面；去黑边实验使用同一批图，未加入其他视频帧。
- 原始证据保存在 251 的 `/home/aigc/wcm-deploy-backups/ocr-repeat-20260909/evaluation/`。可复用脚本：`scripts/eval_ocr_repetition.py`，输入图片、变体 JSON 和输出 JSONL 均须显式指定。
- 此前诊断的四个采样时间相同，但提取/编码方式可能不同；本次各提示、惩罚变体之间使用相同图片。单次顺序测试受缓存、负载影响，耗时不能代表整体视频提速比例。

## 1. 提示模板与结束条件

当前多行系统提示包括“不翻译、不解释、不重复”“≤500字符”“无文字输出空字符串”。本次它在 10 秒帧上只输出换行，在 74 秒帧上连续生成换行直至 300 tokens。图片中实际有招牌和标语，不能把这种循环清理后的空串视为识别成功。

固定 `OCR:`、不带系统提示时，四帧均自然结束，但纯白图凭空识别出 `2023`。因此未直接采用完全无系统提示。

部署选择为图片 + 固定 `OCR:`，系统提示仅保留：

```markdown
## 输出要求
- **输出不超过500个字符**。
```

存储的多行提示仍保留 Markdown 和源代码缩进，在请求构造时 dedent。实际字符上限由应用保证，不依赖模型数数；300 tokens 与 500 字符不作固定换算。

官方模型 `generation_config.json` 的 EOS 是 2。当前服务多数样本返回 `finish_reason=stop`，未观察到所有请求都无法结束。携带系统提示比纯 OCR 提示多 65 个 prompt tokens，表明不能简单认为系统提示全部被忽略。`return_prompt_text` 没有返回渲染文本，实际 tokenizer/部署配置无法通过当前网关核验，未盲目覆盖 EOS。

## 2. 重复惩罚与服务端早停

对比 1.0 / 1.05 / 1.10：旧提示的 74 秒帧全部截断；0、10、580 秒及空白图输出基本一致，未证明有稳定收益。

支持性探测中，`repetition_penalty=-1` 和无效的 `repetition_detection` 均返回 200；无效 `temperature=-1` 返回 400。服务没有对重复参数做预期验证，可能存在网关过滤或后端忽略，不能以 HTTP 200 宣称参数生效。最终请求不加入未经证实的重复惩罚/服务端早停字段。

## 3. 应用侧流式早停

- 一个 OCR 请求使用 SSE 流式响应，不增加重试；网关若返回普通 JSON，则在同一次响应上兼容解析。
- 对原始输出检测持续的精确重复，涵盖短字符循环、长段落和多行循环；短重复及三四块相同标语不因重复直接删除。
- 达到 500 字符或检测到循环即关闭响应流，保留前缀并记录 WARNING，日志不输出识别文字。
- 流中断但已有文字时保留部分文字；任务取消仍传播并关闭响应。正常空结果跳过 Guard。
- 纯换行循环或提前结束后没有有效文字，记录单帧“审核未完成”，不将其变成正常无文字结果，也不取消整个视频任务。
- JSON 兼容路径也使用同样的重复清理和严格 500 字符上限；响应全文返回后清理不能节省生成耗时。

用旧提示在 74 秒帧复现循环：原请求约 1.56 秒、300 输出 tokens；流式候选在收到 40 个换行字符时关闭，约 0.234 秒返回。确认了客户端提前返回和连接关闭；模型服务由外部网关承载，无法核实上游推理是否同步取消，不能把接收字符减少直接算作服务端 token 节省。

## 4. 图像区域评估

去纯黑边会改变模型输入分辨率和识别行为。本次 10 秒帧部分招牌识别恶化，74 秒帧出现额外段落或丢字；未启用自动裁剪。`Spotting:` 在 74 秒帧生成长串 0，耗尽 300 tokens，也未启用。完整文字区域检测管线需要另外验证检测召回率和总请求成本，本版不新增依赖或按固定字幕区域裁剪。

仅启用保守预检查：解码后所有像素（包括透明度）完全一致，才跳过 OCR。低对比度、单像素变化和解码失败均不按纯色处理。纯白图约 0.007 秒返回空串，无 OCR/Guard 请求。

## 候选镜像实测

| 帧 | 旧多行提示 | 候选版 OCR 耗时 | 候选版结果 |
| --- | --- | --- | --- |
| 0 秒 | 正常字幕 | 0.387 秒 | 识别字幕，Guard Safe |
| 10 秒 | 仅换行 | 0.173 秒 | 识别招牌文字，Guard Safe |
| 74 秒 | 300 tokens 换行循环，约 1.56 秒 | 0.333 秒 | 识别标语，Guard 判政治敏感内容 |
| 580 秒 | 正常片尾文字 | 0.201 秒 | 识别片尾文字，Guard Safe |
| 纯白图 | 空结果；纯 OCR 提示则产生幻觉文字 | 0.007 秒 | 本地跳过，零模型请求 |

这些结果证明循环防护和审核通路可用，不代表文字识别完全准确。招牌中的小字、74 秒标语中的中文仍有错字；Guard 命中也不等同于逐字准确率验证。

## 依据

- [PaddleOCR 官方使用教程](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [PaddleOCR-VL 1.6 模型卡](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [官方聊天模板](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/chat_template.jinja)
- [官方 EOS 配置](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/blob/main/generation_config.json)
- [vLLM 重复检测请求定义](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py)
