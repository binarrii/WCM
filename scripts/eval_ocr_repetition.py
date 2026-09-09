"""Bounded, sequential OCR evaluation; only sends explicitly supplied images.

Run inside the deployment environment so credentials stay in existing settings.
Variants are a JSON object mapping names to Chat Completion payload overrides.
Raw OCR evidence is written to --output, never to application logs.
"""

import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx

from wcm_facerec.config import settings


async def evaluate(args):
    variants = json.loads(Path(args.variants).read_text())
    async with httpx.AsyncClient(timeout=60) as client:
        with Path(args.output).open("a") as output:
            for image in args.images:
                encoded = base64.b64encode(Path(image).read_bytes()).decode()
                for name, options in variants.items():
                    options = dict(options)
                    system = options.pop("system", None)
                    prompt = options.pop("prompt", "OCR:")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    if system is not None:
                        messages.insert(0, {"role": "system", "content": system})
                    payload = {
                        "model": "WasuAI/PaddleOCR-VL-1.6",
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0,
                        **options,
                    }
                    start = time.monotonic()
                    row = {"image": Path(image).name, "variant": name}
                    try:
                        response = await client.post(
                            settings.model_api_url,
                            json=payload,
                            headers={"Authorization": f"Bearer {settings.model_api_key}"},
                        )
                        row["status"] = response.status_code
                        row["response"] = response.json()
                    except (httpx.HTTPError, ValueError) as exc:
                        row["error"] = type(exc).__name__
                    row["seconds"] = round(time.monotonic() - start, 3)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    data = row.get("response", {})
                    choice = (data.get("choices") or [{}])[0]
                    print(
                        json.dumps(
                            {k: v for k, v in row.items() if k != "response"}
                            | {
                                "finish": choice.get("finish_reason"),
                                "tokens": data.get("usage", {}).get("completion_tokens"),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("images", nargs="+")
    asyncio.run(evaluate(parser.parse_args()))
