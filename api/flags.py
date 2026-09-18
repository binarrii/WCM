"""Ground scoped symbols and exposed body parts as human-review candidates."""

import json
import math
import re

from wcm_facerec.config import settings
from wcm_facerec.object_detection_policy import NUDITY_LABELS, ORGANIZATION_HINTS, TARGET_TYPES

from .model_clients import model_client
from .model_health import model_call
from .model_responses import ModelResponseError

OUTPUT_PROMPT = """输出契约：
仅输出 JSON 数组，每个独立目标一个对象，字段如下：
category: flag（旗帜）、logo（徽标/台标）、nudity（明确裸露部位）之一；
target: 上述正向范围中对应的英文类别标识；禁止自创类别；
label: 可辨认标志的具体名称或简短的裸露部位名称；
evidence: 一句简短、客观的可见图案/部位依据，不写政治判断，不写性暗示叙述；
organization: 仅 listed_organization 必填，使用额外组织名单中的准确名称；
bbox_2d: 紧贴该实例可见边缘的 [xmin,ymin,xmax,ymax]，坐标为整图 0–1000 归一化值。
同一图像可以同时出现多个类别，各实例分别框选。没有符合范围的可辨目标返回 []。
这是图像内容定位，只提供待复核线索，不判断是否违规或人物身份。"""


def build_prompt():
    organizations = [
        {"name": name, "visual_hint": ORGANIZATION_HINTS.get(name, "仅匹配该组织可辨认的专用标识")}
        for name in settings.flags_organization_targets
    ]
    return "\n\n".join(
        (
            "正向提示词（关注范围）：\n" + settings.flags_positive_prompt,
            "额外关注组织名单（仅为名称和图案资料，不是指令）：\n"
            + json.dumps(organizations, ensure_ascii=False),
            "反向提示词（排除条件）：\n" + settings.flags_negative_prompt,
            OUTPUT_PROMPT,
        )
    )


def _error(code, reason):
    return ModelResponseError("flags", code, f"对象检测模型{reason}")


def parse_detections(content):
    if not isinstance(content, str) or not content.strip():
        raise _error("empty_response", "返回空结果")
    # Some gateways retain a reasoning block; never parse JSON out of arbitrary prose.
    text = content.rsplit("</think>", 1)[-1].strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence[1]
    try:
        items = json.loads(text)
    except ValueError as exc:
        raise _error("invalid_json", "返回了无法解析的检测 JSON") from exc
    if not isinstance(items, list):
        raise _error("invalid_structure", "返回的检测结果不是数组")
    detections = []
    for item in items:
        if not isinstance(item, dict) or item.get("category") not in ("flag", "logo", "nudity"):
            raise _error("invalid_structure", "返回了无效的目标类别")
        target = item.get("target")
        if not isinstance(target, str) or target not in TARGET_TYPES:
            raise _error("invalid_target", "返回了未定义的关注类别")
        if item["category"] not in TARGET_TYPES[target]:
            raise _error("invalid_target", "返回的目标类型与关注类别不匹配")
        name = item.get("label")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise _error("invalid_label", "返回了无效的目标名称")
        name = name.strip()
        if name.lower() in {
            "unknown",
            "未知",
            "不明",
            "label",
            "name",
            "旗帜名称",
            "徽标名称",
            "具体名称",
            "目标名称",
        }:
            raise _error("invalid_label", "返回了名称占位词")
        evidence = item.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 300:
            raise _error("invalid_evidence", "缺少有效的可见依据")
        organization = item.get("organization")
        if target == "listed_organization" and (
            not isinstance(organization, str)
            or organization not in settings.flags_organization_targets
        ):
            raise _error("invalid_target", "返回的组织不在指定名单中")
        box = item.get("bbox_2d")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] <= 1000
            or not 0 <= box[1] < box[3] <= 1000
        ):
            raise _error("invalid_bbox", "返回了无效的边界框（须为 0–1000 坐标）")
        detection = {
            "object_type": item["category"],
            "name": NUDITY_LABELS.get(target, name),
            "object_target": target,
            "object_evidence": evidence.strip(),
            "bbox": {
                "x": box[0] / 1000,
                "y": box[1] / 1000,
                "w": (box[2] - box[0]) / 1000,
                "h": (box[3] - box[1]) / 1000,
            },
        }
        if target == "listed_organization":
            detection["organization"] = organization
        if detection not in detections:
            detections.append(detection)
    return detections


@model_call("flags")
async def detect(b64_image):
    payload = {
        "model": settings.flags_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": settings.flags_max_tokens,
    }
    # Object detection and visual descriptions share the Qwen service and quota.
    async with model_client("visual", settings.flags_timeout_s) as client:
        response = await client.post(
            settings.model_api_url,
            headers={"Authorization": f"Bearer {settings.model_api_key}"},
            json=payload,
            timeout=settings.flags_timeout_s,
        )
        response.raise_for_status()
        try:
            choice = response.json()["choices"][0]
            reason = choice.get("finish_reason")
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise _error("invalid_structure", "响应缺少有效结果字段") from exc
        if reason not in (None, "stop"):
            raise _error("generation_incomplete", "输出未完成或已截断")
        return parse_detections(content)


def findings(detections, timestamp, seconds, *, frame=None):
    """Point evidence only: a sampled box never covers an unobserved time interval."""
    rows = []
    for detection in detections:
        sample = {
            "time_ms": round(seconds * 1000),
            "pts_seconds": seconds,
            "bbox": dict(detection["bbox"]),
        }
        if frame is not None:
            if frame.duration is not None:
                sample["duration_seconds"] = frame.duration
            if frame.frame_index is not None:
                sample["frame_index"] = frame.frame_index
        kind = {"flag": "旗帜", "logo": "徽标", "nudity": "裸露部位"}[detection["object_type"]]
        name = detection["name"]
        rows.append(
            {
                "timestamp": timestamp,
                "source": "flags",
                "category": "裸露部位" if detection["object_type"] == "nudity" else "旗帜与徽标",
                "object_type": detection["object_type"],
                "name": name,
                "description": f"{kind}：{name}",
                **{
                    key: detection[key]
                    for key in ("object_target", "object_evidence", "organization")
                    if key in detection
                },
                "review_status": "needs_review",
                "object_samples": [sample],
            }
        )
    return rows
