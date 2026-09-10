import base64
import io
import json

import httpx
import pytest
from PIL import Image

from api import handlers, ocr


def picture(image):
    data = io.BytesIO()
    image.save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()


def test_uniform_preflight_preserves_even_one_low_contrast_pixel():
    image = Image.new("RGB", (30, 30), "white")
    assert ocr.is_uniform_image(picture(image))
    image.putpixel((15, 15), (254, 255, 255))
    assert not ocr.is_uniform_image(picture(image))
    assert not ocr.is_uniform_image("invalid")
    alpha = Image.new("RGBA", (30, 30), "white")
    alpha.putpixel((15, 15), (255, 255, 255, 0))
    assert not ocr.is_uniform_image(picture(alpha))


@pytest.mark.parametrize("text", ["2023000000", "哈哈哈哈哈", ("同一标语\n" * 4), "a\n\nb\n\nc"])
def test_legitimate_short_repetitions_are_preserved(text):
    assert ocr.clean_output(text) == (text.strip(), False)


@pytest.mark.parametrize(
    "unit", ["\n", "0", "重复", "长段落包含不同的字符并且跨越多行\n下一行也应当一起识别\n"]
)
def test_sustained_loops_keep_prefix_and_warn_without_text_leak(unit, caplog):
    result, repeated = ocr.clean_output("private-prefix\n" + unit * 50)
    assert repeated
    assert result.startswith("private-prefix")
    assert len(result) < len("private-prefix\n" + unit * 50)
    assert "repetition detected" in caplog.text
    assert "private-prefix" not in caplog.text


def test_output_limit_includes_every_character():
    value = "".join(chr(0x4E00 + i) for i in range(700))
    assert ocr.clean_output(value) == (value[:500], False)


class Stream(httpx.AsyncByteStream):
    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.read = 0
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            self.read += 1
            yield event
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def chunk(text=None, finish=None):
    data = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}
    return ("data: " + json.dumps(data) + "\n\n").encode()


async def request(stream):
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        return await ocr.request_ocr(client, "https://ocr.test", {}, {"max_tokens": 300})


@pytest.mark.asyncio
async def test_stream_stops_reading_loop_and_closes_connection():
    stream = Stream([chunk("字幕\n")] + [chunk("重复长句123\n")] * 20 + [chunk("later")])
    response = await request(stream)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    assert choice["ocr_stop_reason"] == "repetition"
    assert stream.read < 10
    assert stream.closed
    assert ocr.clean_output(choice["message"]["content"])[0].startswith("字幕")


@pytest.mark.asyncio
async def test_stream_character_limit_closes_before_next_chunk():
    text = "".join(chr(0x4E00 + i) for i in range(500))
    stream = Stream([chunk(text), chunk("must not be read")])
    response = await request(stream)
    assert stream.read == 1 and stream.closed
    assert response.json()["choices"][0]["ocr_stop_reason"] == "character_limit"


@pytest.mark.asyncio
async def test_normal_empty_stream_remains_valid():
    stream = Stream([chunk(""), chunk(finish="stop")])
    response = await request(stream)
    assert handlers._model_response_text(response, "ocr", 300, allow_empty=True) == ""


@pytest.mark.asyncio
async def test_partial_stream_network_failure_propagates_for_retry():
    stream = Stream([chunk("private subtitle")], httpx.ReadTimeout("private service detail"))
    with pytest.raises(httpx.ReadTimeout):
        await request(stream)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [True, False])
async def test_handler_retries_broken_stream_once(monkeypatch, recover):
    client_class = httpx.AsyncClient
    streams = [
        Stream([chunk("private partial")], httpx.ReadTimeout("private upstream")),
        Stream([chunk("complete text"), chunk(finish="stop")])
        if recover
        else Stream([chunk("private partial")], httpx.ReadTimeout("private upstream")),
    ]
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=streams[len(calls) - 1]
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        handlers.httpx, "AsyncClient", lambda **kwargs: client_class(transport=transport)
    )
    if recover:
        assert await handlers._call_ocr_api("fixture") == "complete text"
    else:
        with pytest.raises(httpx.ReadTimeout):
            await handlers._call_ocr_api("fixture")
    assert len(calls) == 2
    assert all(stream.closed for stream in streams)


@pytest.mark.asyncio
@pytest.mark.parametrize("events", [[], [b"data: [DONE]\n\n"], [chunk("\n")], [chunk("partial")]])
async def test_unfinished_stream_is_not_successful(events):
    with pytest.raises(ValueError):
        await request(Stream(events))


@pytest.mark.asyncio
async def test_cancel_propagates_and_closes_response():
    import asyncio

    stream = Stream([chunk("partial")], asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await request(stream)
    assert stream.closed


@pytest.mark.asyncio
async def test_whitespace_loop_is_a_frame_gap_not_safe_or_task_failure(monkeypatch):
    from tests.test_review_resilience import install_response

    install_response(
        monkeypatch, {"choices": [{"message": {"content": "\n" * 300}, "finish_reason": "length"}]}
    )
    errors = []
    assert (
        await handlers._review_stage("ocr", 74, lambda: handlers._call_ocr_api("fixture"), errors)
        is None
    )
    assert errors[0]["error_code"] == "repetition_without_text"


@pytest.mark.asyncio
async def test_exact_blank_never_calls_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Uniform image must not reach model")

    monkeypatch.setattr(handlers.httpx, "AsyncClient", forbidden)
    assert await handlers._call_ocr_api(picture(Image.new("RGB", (30, 30), "white"))) == ""


@pytest.mark.asyncio
async def test_handler_accepts_real_sse_transport(monkeypatch):
    client_class = httpx.AsyncClient
    stream = Stream([chunk("招牌文字"), chunk(finish="stop")])
    transport = httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    monkeypatch.setattr(
        handlers.httpx, "AsyncClient", lambda **kw: client_class(transport=transport)
    )
    assert await handlers._call_ocr_api("fixture") == "招牌文字"
    assert stream.closed


@pytest.mark.asyncio
async def test_early_stop_with_only_location_data_is_not_safe(monkeypatch):
    from tests.test_review_resilience import install_response

    install_response(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {"content": "<|LOC_0|>"},
                    "finish_reason": "stop",
                    "ocr_stop_reason": "character_limit",
                }
            ]
        },
    )
    errors = []
    await handlers._review_stage("ocr", 74, lambda: handlers._call_ocr_api("fixture"), errors)
    assert errors[0]["error_code"] == "generation_incomplete"
