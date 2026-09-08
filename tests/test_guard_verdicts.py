"""Guard verdicts that require review must reach the saved findings."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from api import handlers


def install_guard_response(monkeypatch, content):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"choices": [{"message": {"content": content}}]},
            )

    monkeypatch.setattr(handlers.httpx, "AsyncClient", lambda **kwargs: Client())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,expected",
    [
        ("Safety: Safe", {"safe": True, "category": ""}),
        (
            "Safety: Unsafe\nCategories: Sexual Content or Sexual Acts",
            {"safe": False, "category": "色情或性行为"},
        ),
        (
            "Safety: Controversial\nCategories: Sexual Content or Sexual Acts",
            {"safe": False, "category": "色情或性行为"},
        ),
        (
            "  SAFETY : cOnTrOvErSiAl  \nCategories: 裸露",
            {"safe": False, "category": "裸露"},
        ),
        ("Safety: Controversial", {"safe": False, "category": "未知敏感内容"}),
        ("Safety: Unsafe", {"safe": False, "category": "未知敏感内容"}),
        (
            "<think>Safety: Controversial</think>\nSafety: Safe",
            {"safe": True, "category": ""},
        ),
        (
            "Safety: Safe\nExplanation: the word controversial is not a verdict here",
            {"safe": True, "category": ""},
        ),
    ],
)
async def test_guard_review_verdicts(monkeypatch, response, expected):
    install_guard_response(monkeypatch, response)
    assert await handlers._call_llm_guard("画面描述") == expected


@pytest.mark.asyncio
async def test_controversial_visual_and_ocr_findings_are_not_dropped(monkeypatch):
    install_guard_response(
        monkeypatch, "Safety: Controversial\nCategories: Sexual Content or Sexual Acts"
    )
    monkeypatch.setattr(handlers, "_download_video_safe_sync", lambda *args: None)
    monkeypatch.setattr(handlers, "_extract_video_windows", lambda *args: [((75.0, "frame"),)])
    monkeypatch.setattr(
        handlers, "_call_nsfw_analysis", AsyncMock(return_value="一名女子裸露上身")
    )
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value="待审核文本"))

    result = await handlers._process_detect_nsfw("https://fixture/video.mp4", 1)

    assert result["visual_analysis"] == [
        {"timestamp": 75.0, "confidence": 1.0, "description": "[色情或性行为] 一名女子裸露上身"}
    ]
    assert result["unsafe_text_frames"] == [
        {"timestamp": 75.0, "category": "色情或性行为", "text": "待审核文本"}
    ]
