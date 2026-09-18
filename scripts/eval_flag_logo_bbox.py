"""Standalone Qwen flag/logo grounding probe; never creates a WCM review task.

prepare writes deterministic, programmatically drawn fixtures and exact boxes.
run sends only fixture pixels + a generic prompt, never ground-truth labels/boxes.
summarize validates coordinates, matches boxes, and renders an HTML contact sheet.
"""

import argparse
import asyncio
import base64
import hashlib
import html
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

MODEL = "WasuAI/Qwen3.8-27B-Abliterated"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def symbol(name, size):
    """Code-defined approximations, not official artwork or natural photographs."""
    w, h = size
    im = Image.new("RGBA", (w, h))
    d = ImageDraw.Draw(im)
    if name in {"china", "japan", "france"}:
        d.rectangle((0, 0, w - 1, h - 1), fill="white")
    if name == "china":
        d.rectangle((0, 0, w - 1, h - 1), fill="#de2910")

        def star(cx, cy, radius, angle):
            points = []
            for i in range(10):
                r = radius if i % 2 == 0 else radius * 0.382
                a = angle + i * math.pi / 5
                points.append((cx + r * math.cos(a), cy + r * math.sin(a)))
            d.polygon(points, fill="#ffde00")

        cx, cy = w / 6, h / 4
        star(cx, cy, h * 0.15, -math.pi / 2)
        for x, y in [(10, 2), (12, 4), (12, 7), (10, 9)]:
            sx, sy = x * w / 30, y * h / 20
            star(sx, sy, h * 0.05, math.atan2(cy - sy, cx - sx))
    elif name == "japan":
        r = h * 0.3
        d.ellipse((w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r), fill="#bc002d")
    elif name == "france":
        d.rectangle((0, 0, w // 3 - 1, h - 1), fill="#002654")
        d.rectangle((2 * w // 3, 0, w - 1, h - 1), fill="#ed2939")
    elif name == "youtube":
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h * 0.23, fill="#ff0000")
        d.polygon([(w * 0.4, h * 0.25), (w * 0.4, h * 0.75), (w * 0.7, h * 0.5)], fill="white")
    elif name == "olympic":
        # Simplified overlapping rings; interlacing is not reproduced.
        r = w / 6.5
        for cx, cy, color in [
            (r, r, "#0081c8"),
            (3.25 * r, r, "black"),
            (5.5 * r, r, "#ee334e"),
            (2.125 * r, h - r, "#fcb131"),
            (4.375 * r, h - r, "#00a651"),
        ]:
            d.ellipse(
                (cx - r, cy - r, cx + r - 1, cy + r - 1),
                outline=color,
                width=max(2, round(r * 0.19)),
            )
    elif name == "mercedes":
        width = max(2, round(w * 0.04))
        d.ellipse((0, 0, w - 1, h - 1), outline="#20252b", width=width)
        for angle in [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]:
            d.polygon(
                [
                    (w / 2 + 0.48 * w * math.cos(angle), h / 2 + 0.48 * h * math.sin(angle)),
                    (
                        w / 2 + 0.08 * w * math.cos(angle + math.pi / 2),
                        h / 2 + 0.08 * h * math.sin(angle + math.pi / 2),
                    ),
                    (
                        w / 2 + 0.08 * w * math.cos(angle - math.pi / 2),
                        h / 2 + 0.08 * h * math.sin(angle - math.pi / 2),
                    ),
                ],
                fill="#20252b",
            )
    else:
        raise ValueError(name)
    return im


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    specs = [
        ("single_flag", (900, 600), [("china", 177, 131, 360, 240)]),
        ("single_logo", (900, 600), [("youtube", 453, 347, 240, 135)]),
        (
            "three_flags",
            (1200, 700),
            [
                ("china", 60, 80, 300, 200),
                ("japan", 450, 270, 270, 180),
                ("france", 855, 460, 270, 180),
            ],
        ),
        (
            "three_logos",
            (1100, 660),
            [
                ("youtube", 60, 90, 240, 135),
                ("olympic", 430, 270, 260, 120),
                ("mercedes", 830, 430, 170, 170),
            ],
        ),
        (
            "mixed_wide",
            (1280, 720),
            [
                ("china", 80, 60, 240, 160),
                ("japan", 930, 70, 210, 140),
                ("olympic", 510, 290, 260, 120),
                ("youtube", 1050, 590, 160, 90),
            ],
        ),
        (
            "repeated_objects",
            (1000, 700),
            [
                ("japan", 55, 65, 210, 140),
                ("japan", 400, 65, 210, 140),
                ("japan", 750, 65, 210, 140),
                ("youtube", 140, 450, 180, 102),
                ("youtube", 650, 420, 240, 135),
            ],
        ),
        (
            "small_objects",
            (1280, 720),
            [
                ("china", 90, 120, 66, 44),
                ("japan", 620, 320, 54, 36),
                ("youtube", 1090, 580, 48, 27),
                ("olympic", 870, 160, 91, 42),
            ],
        ),
        (
            "edge_objects",
            (900, 610),
            [
                ("france", 0, 30, 150, 100),
                ("mercedes", 790, 500, 110, 110),
                ("youtube", 640, 0, 180, 102),
            ],
        ),
        (
            "portrait",
            (600, 1000),
            [
                ("china", 75, 85, 240, 160),
                ("youtube", 320, 470, 160, 90),
                ("olympic", 105, 830, 260, 120),
            ],
        ),
        ("blank", (900, 600), []),
        ("non_target_shapes", (900, 600), []),
    ]
    cases = []
    for name, size, objects in specs:
        im = Image.new("RGB", size, "#dce3e9")
        truth = []
        for label, x, y, w, h in objects:
            asset = symbol(label, (w, h))
            im.paste(asset, (x, y), asset)
            truth.append(
                {
                    "category": "flag" if label in {"china", "japan", "france"} else "logo",
                    "label": label,
                    "bbox": [x, y, x + w, y + h],
                }
            )
        if name == "non_target_shapes":
            d = ImageDraw.Draw(im)
            d.ellipse((70, 80, 210, 220), fill="#476dd6")
            d.polygon([(440, 130), (340, 310), (540, 310)], fill="#ff9800")
            d.line([(610, 450), (710, 340), (810, 470)], fill="#70449d", width=18)
        filename = name + ".png"
        im.save(folder / filename)
        cases.append(
            {
                "id": name,
                "image": filename,
                "size": list(size),
                "sha256": hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                "objects": truth,
            }
        )
    write_json(
        folder / "manifest.json",
        {"kind": "programmatic synthetic fixtures; approximate artwork", "cases": cases},
    )


def prompt(size, coordinates, style="explicit"):
    if style == "compact":
        if coordinates != "normalized":
            raise ValueError("compact prompt only supports normalized coordinates")
        return (
            "请检测并定位图片中的所有旗帜、品牌标志和组织徽标。每个独立目标一个元素，"
            "重复出现的目标分别输出，旗帜包含完整旗面，徽标包含完整图案，普通几何图形不算徽标。"
            "以 JSON 数组输出，每项格式为 "
            '{"category":"flag 或 logo","label":"名称","bbox_2d":[x1,y1,x2,y2]}。'
            "bbox_2d 为 0–1000 的归一化坐标，表示左上角和右下角；不要输出像素坐标。"
            "没有目标时输出 []，名称不确定时写 unknown。仅输出 JSON。"
        )
    w, h = size
    units = (
        "0 到 1000 的归一化坐标；x 除以图片宽度后乘 1000，y 除以图片高度后乘 1000"
        if coordinates == "normalized"
        else f"原图像素坐标；x 范围为 0 到 {w}，y 范围为 0 到 {h}，不要使用归一化坐标"
    )
    return (
        f"检测并定位图片中所有可见的旗帜、品牌标志和组织徽标。原图宽 {w} 像素、高 {h} 像素。"
        "每个独立出现的目标输出一个框，重复出现的标志分别输出；不要把一个徽标拆成多个元素。"
        "旗帜框覆盖完整旗面，徽标框紧贴完整图案，不包括周围空白。普通几何图形不要当作徽标。"
        "没有目标时输出空数组 []。名称无法确定时写 unknown，不要猜测。"
        f"bbox_2d 必须使用{units}，顺序为 [xmin,ymin,xmax,ymax]，原点为左上角。"
        '仅输出 JSON 数组，每项格式为 {"category":"flag 或 logo","label":"目标名称","bbox_2d":[xmin,ymin,xmax,ymax]}。'
        "不要输出思考过程、Markdown 或解释。"
    )


async def run(args):
    from api import parameter_store
    from wcm_facerec.cluster import model_slot
    from wcm_facerec.config import settings

    if settings.review_tasks_db_enabled:
        await parameter_store.refresh()  # Read existing gateway config; no schema/init/write.
    cases = json.loads(args.manifest.read_text())["cases"]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases.split(",")]
    # One outstanding request, uses the deployed visual admission limit, no retries.
    async with httpx.AsyncClient(timeout=60) as client:
        for case in cases:
            raw = (args.manifest.parent / case["image"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == case["sha256"]
            instruction = (
                args.prompt_file.read_text()
                if args.prompt_file
                else prompt(case["size"], args.coordinates, args.prompt_style)
            )
            record = {
                "case": case["id"],
                "coordinates": args.coordinates,
                "prompt_style": "custom-" + args.prompt_file.stem
                if args.prompt_file
                else args.prompt_style,
                "model": MODEL,
                "sha256": case["sha256"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": instruction,
                "temperature": 0,
                "max_tokens": 2048,
            }
            started = time.monotonic()
            try:
                async with model_slot("visual"):
                    admitted = time.monotonic()
                    response = await client.post(
                        settings.model_api_url,
                        headers={"Authorization": "Bearer " + settings.model_api_key},
                        json={
                            "model": MODEL,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": "data:image/png;base64,"
                                                + base64.b64encode(raw).decode()
                                            },
                                        },
                                        {"type": "text", "text": instruction},
                                    ],
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 2048,
                        },
                    )
                    record.update(
                        status=response.status_code,
                        seconds=round(time.monotonic() - admitted, 3),
                        queue_seconds=round(admitted - started, 3),
                    )
                    if response.is_success:
                        data = response.json()
                        choice = data["choices"][0]
                        record.update(
                            text=choice["message"].get("content"),
                            finish_reason=choice.get("finish_reason"),
                            usage=data.get("usage"),
                            response_model=data.get("model"),
                        )
                    else:
                        record["error"] = f"HTTP {response.status_code}"
            except Exception as exc:
                # Avoid logging a URL, bearer token or request body from exceptions.
                record.update(
                    error=type(exc).__name__, seconds=round(time.monotonic() - started, 3)
                )
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if record.get("status") in {401, 403, 429} or "error" in record:
                break


def parse_boxes(record, size):
    if record.get("finish_reason") != "stop" or record.get("error"):
        raise ValueError("request failed or generation incomplete")
    text = record["text"].strip()
    strict_json = not text.startswith("```")
    if not strict_json:
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("unclosed code fence")
        text = "\n".join(lines[1:-1])
    values = json.loads(text)
    if not isinstance(values, list):
        raise ValueError("response must be an array")
    w, h = size
    limits = [1000, 1000, 1000, 1000] if record["coordinates"] == "normalized" else [w, h, w, h]
    boxes = []
    for item in values:
        b = item["bbox_2d"]
        if (
            not isinstance(b, list)
            or len(b) != 4
            or any(type(v) not in {int, float} or not math.isfinite(v) for v in b)
        ):
            raise ValueError("bbox must contain four finite numbers")
        if (
            any(v < 0 or v > limit for v, limit in zip(b, limits, strict=True))
            or b[0] >= b[2]
            or b[1] >= b[3]
        ):
            raise ValueError("out-of-range or inverted bbox")
        if item.get("category") not in {"flag", "logo"} or not isinstance(item.get("label"), str):
            raise ValueError("invalid category/label")
        pixels = [
            v * scale / limit for v, scale, limit in zip(b, [w, h, w, h], limits, strict=True)
        ]
        boxes.append({**item, "bbox": pixels})
    return boxes, strict_json


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


def score(truth, predictions, threshold=0.5):
    # Maximum-cardinality bipartite matching: a repeated prediction cannot match twice.
    edges = [
        [
            j
            for j, p in sorted(
                enumerate(predictions), key=lambda pair: -iou(g["bbox"], pair[1]["bbox"])
            )
            if p["category"] == g["category"] and iou(g["bbox"], p["bbox"]) >= threshold
        ]
        for g in truth
    ]
    matched = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(truth)):
        augment(i, set())
    best = [
        max(
            (iou(g["bbox"], p["bbox"]) for p in predictions if p["category"] == g["category"]),
            default=0,
        )
        for g in truth
    ]
    return {
        "tp": len(matched),
        "fp": len(predictions) - len(matched),
        "fn": len(truth) - len(matched),
        "best_gt_iou": best,
        "matches": [
            {"gt": i, "prediction": j, "iou": iou(truth[i]["bbox"], predictions[j]["bbox"])}
            for j, i in matched.items()
        ],
    }


def summarize(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    cases = {c["id"]: c for c in manifest["cases"]}
    description = html.escape(manifest.get("description", "程序绘制的受控样本"))
    records = [
        json.loads(line)
        for path in sorted(folder.glob("responses-*.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    results, cards = [], []
    for record in records:
        case = cases[record["case"]]
        variant = record["coordinates"]
        if record.get("prompt_style", "explicit") != "explicit":
            variant += "-" + record["prompt_style"]
        predictions, strict = [], False
        error = None
        try:
            predictions, strict = parse_boxes(record, case["size"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            error = str(exc)
        scores = score(case["objects"], predictions)
        result = {
            "case": case["id"],
            "coordinates": record["coordinates"],
            "variant": variant,
            "seconds": record["seconds"],
            "valid": error is None,
            "strict_json": strict,
            "error": error,
            "predictions": predictions,
            **scores,
        }
        results.append(result)
        im = Image.open(folder / case["image"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for g in case["objects"]:
            d.rectangle(g["bbox"], outline="#128a3e", width=4)
        for index, p in enumerate(predictions):
            d.rectangle(p["bbox"], outline="#d52683", width=2)
            x, y = p["bbox"][:2]
            d.text((x + 4, max(0, y - 18)), str(index + 1), fill="#b31265", font_size=18)
        filename = f"overlay-{case['id']}-{variant}.png"
        im.save(folder / filename)
        cards.append(
            f"<article><h2>{html.escape(case['id'])} · {variant}</h2><p>TP {scores['tp']} / FP {scores['fp']} / FN {scores['fn']} · {record['seconds']} s</p><img src='{filename}'><pre>{html.escape(record.get('text') or str(error))}</pre></article>"
        )
    totals = {}
    for mode in sorted({r["variant"] for r in results}):
        subset = [r for r in results if r["variant"] == mode]
        total = {
            key: sum(r[key] for r in subset) for key in ("tp", "fp", "fn", "valid", "strict_json")
        }
        best = [v for r in subset for v in r["best_gt_iou"]]
        total.update(
            requests=len(subset),
            precision=total["tp"] / (total["tp"] + total["fp"])
            if total["tp"] + total["fp"]
            else None,
            recall=total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else None,
            mean_best_gt_iou=statistics.mean(best) if best else None,
            median_seconds=statistics.median(r["seconds"] for r in subset),
        )
        totals[mode] = total
    write_json(
        folder / "summary.json",
        {
            "matching": "category-aware maximum cardinality at IoU >= 0.5; specific names reviewed separately",
            "totals": totals,
            "results": results,
        },
    )
    (folder / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Qwen flag/logo bbox probe</title><style>body{font:16px system-ui;margin:28px;background:#f4f6f8;color:#18232d}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:20px}article{background:white;padding:18px;border-radius:12px}img{width:100%}pre{white-space:pre-wrap;font-size:13px}h2{font-size:18px}</style><h1>Qwen · 旗帜与徽标 bbox 实测</h1><p>"
        + description
        + "；绿色：参考框，粉色：模型框。名称识别与框定位分开评估。仅验证能力，不代表真实视频准确率。</p><pre>"
        + html.escape(json.dumps(totals, ensure_ascii=False, indent=2))
        + "</pre><main>"
        + "".join(cards)
        + "</main>"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "summarize"):
        commands.add_parser(name).add_argument("folder", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument("--coordinates", choices=["normalized", "pixels"], default="normalized")
    runner.add_argument("--prompt-style", choices=["explicit", "compact"], default="explicit")
    runner.add_argument(
        "--prompt-file", type=Path, help="Use a custom prompt; saved verbatim in responses"
    )
    runner.add_argument("--cases", help="Comma-separated case IDs; defaults to all")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.folder)
    elif args.command == "summarize":
        summarize(args.folder)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
