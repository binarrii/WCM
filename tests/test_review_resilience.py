"""Frame failures preserve sibling findings, later frames and explicit review gaps."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from api import handlers, utils
from tests.test_video_windows import Capture, pixel


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["face", "visual", "ocr"])
async def test_one_failed_frame_preserves_siblings_and_later_frames(monkeypatch, failed_stage):
    cap = Capture(3)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *a, **kw: None)
    seen = {stage: [] for stage in ("face", "visual", "ocr")}

    def visit(stage, index):
        seen[stage].append(index)
        if stage == failed_stage and index == 1:
            raise httpx.ReadTimeout("private response must not appear in results")

    async def face(**kwargs):
        index = int(kwargs["img_source"][0, 0, 0])
        visit("face", index)
        return {"all_results": [{"category": "人物", "name": f"person-{index}"}]}

    async def visual(images, times):
        index = pixel(images[0])
        visit("visual", index)
        return f"visual-{index}"

    async def ocr(image):
        index = pixel(image)
        visit("ocr", index)
        return f"ocr-{index}"

    monkeypatch.setattr(
        handlers, "get_face_engine", lambda: SimpleNamespace(search_multi_face=face)
    )
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", visual)
    monkeypatch.setattr(handlers, "_call_ocr_api", ocr)
    monkeypatch.setattr(
        handlers, "_call_llm_guard", AsyncMock(return_value={"safe": False, "category": "复核"})
    )

    result = await asyncio.wait_for(
        handlers._process_analyze_media("http://test/video.mp4", 0.5, 5, 0.5), 5
    )
    gaps = [row for row in result if row.get("review_status") == "incomplete"]
    assert len(gaps) == 1
    assert gaps[0]["stage"] == failed_stage
    assert gaps[0]["timestamp"] == "00:00:00.500"
    assert "超时" in gaps[0]["description"]
    assert "private" not in str(result)
    descriptions = {row["description"] for row in result if row not in gaps}
    expected = {f"{prefix}-{i}" for prefix in ("person", "visual", "ocr") for i in range(3)}
    expected.remove(f"{'person' if failed_stage == 'face' else failed_stage}-1")
    assert descriptions == expected
    assert all(sorted(indices) == [0, 1, 2] for indices in seen.values())
    assert cap.released


def install_response(monkeypatch, response):
    client = AsyncMock()
    client.__aenter__.return_value = client
    if isinstance(response, Exception):
        client.post.side_effect = response
    else:
        client.post.return_value = httpx.Response(
            200, json=response, request=httpx.Request("POST", "https://model.test")
        )
    monkeypatch.setattr(handlers.httpx, "AsyncClient", lambda **kw: client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,finish",
    [("", "stop"), ("cannot answer", "stop"), (None, "stop"), ("Safety: Safe", "length")],
)
async def test_invalid_guard_verdict_becomes_incomplete(monkeypatch, content, finish):
    install_response(
        monkeypatch, {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    )
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value="text to audit"))
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    result = await handlers._process_detect_sensitive("http://test/image.jpg", 1)
    assert result["unsafe_text_frames"] == []
    assert result["errors"][0]["review_status"] == "incomplete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.ReadTimeout("timeout"),
        {},
        {"choices": [{"message": {"content": "text"}, "finish_reason": "length"}]},
    ],
)
async def test_ocr_errors_are_recorded_without_retry(monkeypatch, response):
    client = install_response(monkeypatch, response)
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    result = await handlers._process_detect_sensitive("http://test/image.jpg", 1)
    assert result["unsafe_text_frames"] == []
    assert result["errors"][0]["stage"] == "ocr"
    client.post.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["", " \n\t ", "<|LOC_0|> <|LOC_1|>"])
async def test_valid_empty_ocr_is_not_an_error(monkeypatch, content):
    client = install_response(
        monkeypatch, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    )
    guard = AsyncMock()
    monkeypatch.setattr(handlers, "_call_llm_guard", guard)
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    assert await handlers._process_detect_sensitive("http://test/image.jpg", 1) == {
        "unsafe_text_frames": []
    }
    guard.assert_not_awaited()
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_ocr_uses_recognition_task_prompt_and_preserves_text(monkeypatch):
    client = install_response(
        monkeypatch,
        {
            "choices": [
                {"message": {"content": "测试字幕\nHello World 123"}, "finish_reason": "stop"}
            ]
        },
    )
    assert await handlers._call_ocr_api("fixture-image") == "测试字幕\nHello World 123"
    payload = client.post.await_args.kwargs["json"]
    assert payload["max_tokens"] == 1024
    assert payload["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,fixture-image"}},
        {"type": "text", "text": "OCR:"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,component,code,message",
    [
        (
            {"choices": [{"message": {"content": "unfinished"}, "finish_reason": "length"}]},
            "ocr",
            "output_truncated",
            "1024 tokens",
        ),
        ({}, "ocr", "invalid_structure", "缺少有效结果字段"),
        (
            {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
            "ocr",
            "invalid_content",
            "不是文本",
        ),
        (
            {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]},
            "ocr",
            "generation_incomplete",
            "未正常完成",
        ),
        (
            {"choices": [{"message": {"content": "uncertain"}, "finish_reason": "stop"}]},
            "guard",
            "missing_safety_verdict",
            "Safety 判定",
        ),
        (
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            "guard",
            "empty_response",
            "返回空结果",
        ),
        (
            {"choices": [{"message": {"content": "Safety: Safe"}, "finish_reason": "length"}]},
            "guard",
            "output_truncated",
            "512 tokens",
        ),
    ],
)
async def test_response_failures_explain_component_and_reason(
    monkeypatch, response, component, code, message, caplog
):
    client = install_response(monkeypatch, response)
    if component == "guard":
        monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value="ordinary text"))
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    result = await handlers._process_detect_sensitive("http://test/image.jpg", 1)
    assert result["unsafe_text_frames"] == []
    error = result["errors"][0]
    assert error["stage"] == "ocr"
    assert error["component"] == component
    assert error["error_code"] == code
    assert message in error["description"]
    assert "帧数据无效" not in error["description"]
    assert f"component={component} code={code}" in caplog.text
    client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_json_reports_format_problem_without_body_leak(monkeypatch, caplog):
    client = install_response(monkeypatch, {})
    client.post.return_value = httpx.Response(
        200, text="private upstream response", request=httpx.Request("POST", "https://model.test")
    )
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    result = await handlers._process_detect_sensitive("http://test/image.jpg", 1)
    assert result["errors"][0]["error_code"] == "invalid_json"
    assert "private upstream response" not in str(result) + caplog.text


@pytest.mark.asyncio
async def test_standalone_visual_failure_keeps_text_findings(monkeypatch):
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=b"fixture"))
    monkeypatch.setattr(
        handlers,
        "_call_nsfw_analysis",
        AsyncMock(side_effect=handlers.NsfwAnalysisError("offline")),
    )
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value="text to review"))
    monkeypatch.setattr(
        handlers, "_call_llm_guard", AsyncMock(return_value={"safe": False, "category": "待复核"})
    )
    result = await handlers._process_detect_nsfw("http://test/image.jpg", 1)
    assert result["visual_analysis"] == []
    assert result["unsafe_text_frames"] == [
        {"type": "image", "category": "待复核", "text": "text to review"}
    ]
    assert result["errors"][0]["stage"] == "visual"


@pytest.mark.asyncio
async def test_video_decode_failure_is_still_fatal_and_cleans_up_workers(monkeypatch):
    cap = Capture(100, fail_at=4)
    monkeypatch.setattr(utils.cv2, "VideoCapture", lambda _: cap)
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *a, **kw: None)
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_nsfw_analysis", AsyncMock(return_value="scene"))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_call_llm_guard", AsyncMock(return_value={"safe": True}))
    with pytest.raises(RuntimeError, match="decode failed"):
        await asyncio.wait_for(
            handlers._process_analyze_media("http://test/video.mp4", 0.5, 5, 0.5), 5
        )
    assert cap.released
    assert not [t for t in asyncio.all_tasks() if t.get_coro().__qualname__.endswith(".consumer")]
