"""Bounded OCR output handling, including cancellation of repetitive streams."""

import base64
import binascii
import io
import json
import logging
import re

import httpx
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)
MAX_CHARACTERS = 500


def is_uniform_image(encoded: str) -> bool:
    """Only skip exactly constant pixels, including alpha; never guess contrast."""
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
            return all(low == high for low, high in image.convert("RGBA").getextrema())
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
        # Optional preflight must not turn an undecodable image into a safe result.
        return False


def repetition_cut(text: str) -> int | None:
    """Find sustained exact loops, preserving their first occurrence.

    Short legitimate repetitions, repeated signs and multiline layout survive.
    Detection runs on raw output so whitespace-only loops are not lost to strip().
    """
    cuts = []
    for pattern in (r"([\s\S]{1,3}?)\1{11,}", r"([\s\S]{4,160}?)\1{4,}"):
        for match in re.finditer(pattern, text):
            if len(match[0]) >= 40:
                cuts.append(match.start() + len(match[1]))
    return min(cuts) if cuts else None


def clean_output(text: str) -> tuple[str, bool]:
    cut = repetition_cut(text)
    if cut is not None:
        logger.warning("OCR repetition detected; keeping prefix: received_chars=%s", len(text))
        text = text[:cut]
    text = re.sub(r"<\|LOC_\d+\|>", "", text).strip()
    if len(text) > MAX_CHARACTERS:
        logger.warning("OCR character limit reached; keeping prefix: limit=%s", MAX_CHARACTERS)
    return text[:MAX_CHARACTERS], cut is not None


async def request_ocr(client, url: str, headers: dict, payload: dict) -> httpx.Response:
    """One request only. Closing the context cancels consumption on early stop.

    JSON responses remain compatible with gateways which ignore stream=True.
    Broken streams propagate so the shared model wrapper can retry once.
    """
    async with client.stream(
        "POST", url, headers=headers, json={**payload, "stream": True}
    ) as response:
        response.raise_for_status()
        if "text/event-stream" not in response.headers.get("content-type", ""):
            await response.aread()
            return response
        text = ""
        finish = None
        stopped = None
        event_lines = []

        async def events():
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip(" "))
                elif not line and event_lines:
                    yield "\n".join(event_lines)
                    event_lines.clear()
            if event_lines:
                yield "\n".join(event_lines)

        async for event in events():
            if event == "[DONE]":
                break
            data = json.loads(event)
            if data.get("error"):
                raise ValueError("OCR stream reported an upstream error")
            choices = data.get("choices")
            if choices == []:  # Optional final usage event.
                continue
            if not isinstance(choices, list) or not choices:
                raise ValueError("OCR stream has no choices")
            choice = choices[0]
            delta = choice.get("delta", {}).get("content")
            if delta is not None:
                if not isinstance(delta, str):
                    raise ValueError("OCR stream content is not text")
                text += delta
            finish = choice.get("finish_reason") or finish
            if repetition_cut(text) is not None:
                stopped = "repetition"
                break
            if len(text) >= MAX_CHARACTERS:
                stopped = "character_limit"
                break
            if finish is not None:
                break
        if finish is None and stopped is None:
            raise ValueError("OCR stream ended without completion")
        if stopped:
            logger.warning(
                "OCR stream stopped; keeping partial content: reason=%s received_chars=%s",
                stopped,
                len(text),
            )
        return httpx.Response(
            200,
            request=response.request,
            json={
                "choices": [
                    {
                        "message": {"content": text},
                        "finish_reason": finish or "stop",
                        "ocr_stop_reason": stopped,
                    }
                ]
            },
        )
