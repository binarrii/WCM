import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest

from api import flags, handlers, model_health, review_windows
from api.model_responses import ModelResponseError
from api.review_coverage import ReviewCoverage
from api.review_progress import ReviewProgress
from api.review_results import consolidate_results, flatten_findings
from tests.test_window_review import install_video, sample
from wcm_facerec.config import Settings, settings

FLAG = {
    "category": "flag",
    "target": "taiwan_related",
    "label": "中华民国旗",
    "evidence": "红底，左上蓝底白日图案",
    "bbox_2d": [510, 80, 565, 160],
}
LOGO = {
    "category": "logo",
    "target": "listed_organization",
    "organization": "新唐人",
    "label": "新唐人台标",
    "evidence": "金色圆形唐字图案",
    "ntd_identity": {"kind": "tang_emblem", "text": "唐"},
    "bbox_2d": [850, 20, 950, 180],
}
NUDITY = {
    "category": "nudity",
    "target": "exposed_female_nipple",
    "label": "裸露乳头",
    "evidence": "乳房区域乳头及乳晕实际可见，未被遮挡",
    "bbox_2d": [100, 200, 160, 260],
}
MAP = {
    "category": "map",
    "target": "china_map_missing_region",
    "label": "中国地图疑似缺少台湾",
    "map_region": "台湾",
    "evidence": "标题为中国全图，东南海岸以东的图幅完整可见但未绘台湾轮廓",
    "bbox_2d": [100, 100, 900, 900],
}


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "<think>reasoning</think>\n{}"])
def test_structured_multiple_targets_and_named_organization(wrapper):
    results = flags.parse_detections(wrapper.format(json.dumps([FLAG, LOGO, FLAG])))
    assert len(results) == 2
    assert results[0] == {
        "object_type": "flag",
        "name": "中华民国旗",
        "object_target": "taiwan_related",
        "object_evidence": FLAG["evidence"],
        "bbox": {"x": 0.51, "y": 0.08, "w": 0.055, "h": 0.08},
    }
    assert results[1]["object_type"] == "logo" and results[1]["organization"] == "新唐人"
    assert flags.parse_detections("[]") == []


@pytest.mark.parametrize(
    "content",
    [
        "",
        "无",
        "[",
        "{}",
        "Here are the flags: []",
        "[null]",
        json.dumps([{**FLAG, "category": "person"}]),
        json.dumps([{**FLAG, "label": "旗帜名称"}]),
        json.dumps([{**FLAG, "label": "unknown"}]),
        json.dumps([{**FLAG, "target": "ordinary_national_flag", "label": "日本国旗"}]),
        json.dumps([{**FLAG, "target": "exposed_genitals"}]),
        json.dumps([{**FLAG, "target": None}]),
        json.dumps([{**FLAG, "evidence": ""}]),
        json.dumps([{**LOGO, "organization": "名单外组织"}]),
        json.dumps([{**NUDITY, "bbox_2d": [0, 0, 0, 0]}]),
        json.dumps([{**FLAG, "bbox_2d": [1, 2, 1001, 4]}]),
        json.dumps([{**FLAG, "bbox_2d": [2, 2, 1, 4]}]),
        json.dumps([{**FLAG, "bbox_2d": [1, 2, True, 4]}]),
        json.dumps([{**FLAG, "bbox_2d": [1, 2, float("nan"), 4]}]),
        json.dumps([FLAG, {**LOGO, "bbox_2d": [1, 2]}]),
    ],
)
def test_invalid_or_partially_invalid_results_are_never_safe(content):
    with pytest.raises(ModelResponseError) as error:
        flags.parse_detections(content)
    assert error.value.component == "flags"


@pytest.mark.parametrize(
    "identity",
    [
        {"kind": "wordmark", "text": "NTD"},
        {"kind": "wordmark", "text": " ntdtv "},
        {"kind": "tang_emblem", "text": "唐"},
    ],
)
def test_ntd_identity_accepts_only_its_readable_marks(identity):
    result = flags.parse_detections(json.dumps([{**LOGO, "ntd_identity": identity}]))
    assert len(result) == 1 and result[0]["organization"] == "新唐人"


def test_rejected_ntd_identity_does_not_discard_other_findings():
    rejected = {
        **LOGO,
        "evidence": "四颗白色星形围合，未见NTD字标或篆体唐字，符合排除特征",
        "ntd_identity": {"kind": "none", "text": ""},
    }
    result = flags.parse_detections(json.dumps([FLAG, rejected, NUDITY, MAP]))
    assert [r["object_type"] for r in result] == ["flag", "nudity", "map"]


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        {"kind": "unknown", "text": ""},
        {"kind": "wordmark", "text": None},
        {"kind": "wordmark", "text": "旅游卫视"},
        {"kind": "wordmark", "text": "未见NTD"},
        {"kind": "wordmark", "text": "唐"},
        {"kind": "tang_emblem", "text": "NTD"},
        {"kind": "tang_emblem", "text": ""},
    ],
)
def test_invalid_ntd_identity_is_not_silently_treated_as_safe(identity):
    with pytest.raises(ModelResponseError) as error:
        flags.parse_detections(json.dumps([{**LOGO, "ntd_identity": identity}]))
    assert error.value.code == "invalid_identity"


def install_detector(monkeypatch, *, content=None, finish="stop"):
    calls, slots = [], []

    def respond(request):
        payload = json.loads(request.content)
        payload["_request_url"] = str(request.url)
        payload["_authorization"] = request.headers.get("authorization")
        calls.append(payload)
        assert request.extensions["timeout"]["read"] == settings.flags_timeout_s
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {
                            "content": content if content is not None else json.dumps([FLAG, LOGO])
                        },
                    }
                ]
            },
        )

    @asynccontextmanager
    async def client(model, timeout):
        slots.append(model)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as connection:
            yield connection

    monkeypatch.setattr(flags, "model_client", client)
    return calls, slots


@pytest.mark.asyncio
async def test_request_uses_dedicated_endpoint_quota_and_configured_model(monkeypatch):
    monkeypatch.setattr(settings, "flags_timeout_s", 3)
    calls, slots = install_detector(monkeypatch)
    assert len(await flags.detect("image-data")) == 2
    assert slots == ["flags"]
    assert calls[0]["model"] == "Ornith-1.5-35B-A3B"
    assert calls[0]["_request_url"] == "http://10.252.25.217:8800/v1/chat/completions"
    assert calls[0]["_authorization"] is None
    assert calls[0]["max_tokens"] == 2048
    assert calls[0]["reasoning_effort"] == "none"
    assert [part["type"] for part in calls[0]["messages"][0]["content"]] == ["text", "image_url"]


@pytest.mark.asyncio
async def test_detector_base_url_and_key_are_independent_of_visual_service(monkeypatch):
    monkeypatch.setattr(settings, "flags_api_base_url", "https://detector.example/v1/")
    monkeypatch.setattr(settings, "flags_api_key", "detector-secret")
    monkeypatch.setattr(settings, "model_api_url", "https://visual.example/v1/chat/completions")
    monkeypatch.setattr(settings, "model_api_key", "visual-secret")
    calls, slots = install_detector(monkeypatch, content="[]")
    assert await flags.detect("image-data") == []
    assert slots == ["flags"]
    assert calls[0]["_request_url"] == "https://detector.example/v1/chat/completions"
    assert calls[0]["_authorization"] == "Bearer detector-secret"


@pytest.mark.asyncio
async def test_detector_can_omit_reasoning_effort_for_other_services(monkeypatch):
    monkeypatch.setattr(settings, "flags_reasoning_effort", "auto")
    calls, _ = install_detector(monkeypatch, content="[]")
    assert await flags.detect("image-data") == []
    assert "reasoning_effort" not in calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(("effort", "thinking"), [("none", False), ("low", True)])
async def test_detector_falls_back_to_enable_thinking_when_effort_is_unsupported(
    monkeypatch, effort, thinking
):
    monkeypatch.setattr(settings, "flags_reasoning_effort", effort)
    requests, slots = [], []

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={"error": {"code": "unsupported_parameter", "param": "reasoning_effort"}},
            )
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "[]"}}]},
        )

    @asynccontextmanager
    async def client(model, timeout):
        slots.append(model)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as connection:
            yield connection

    monkeypatch.setattr(flags, "model_client", client)
    assert await flags.detect("image-data") == []
    assert slots == ["flags"]
    assert requests[0]["reasoning_effort"] == effort
    assert "chat_template_kwargs" not in requests[0]
    assert "reasoning_effort" not in requests[1]
    assert requests[1]["chat_template_kwargs"] == {"enable_thinking": thinking}


@pytest.mark.asyncio
@pytest.mark.parametrize("organizations", [["新唐人", "新中国联邦"], []])
async def test_organization_rules_reach_request_with_existing_stored_prompts(
    monkeypatch, organizations
):
    from wcm_facerec import runtime_parameters

    # Existing deployments freeze DB prompts; changing defaults alone would leave
    # the overly broad small-star instruction active in actual model requests.
    legacy_negative = "不要仅因目标小或星点细节不清就排除"
    configured = {
        "flags_positive_prompt": "已有自定义关注范围",
        "flags_negative_prompt": legacy_negative,
        "flags_organization_targets": organizations,
    }
    calls, slots = install_detector(monkeypatch, content="[]")
    with runtime_parameters.frozen(configured):
        assert await flags.detect("original-unannotated-frame") == []
        assert settings.flags_negative_prompt == legacy_negative
        assert settings.flags_organization_targets == organizations

    assert len(calls) == 1 and slots == ["flags"]
    content = calls[0]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].endswith("original-unannotated-frame")
    prompt = content[0]["text"]
    assert "已有自定义关注范围" in prompt and legacy_negative in prompt
    if organizations:
        assert prompt.index("组织标识严格匹配") > prompt.index(legacy_negative)
        assert "反例：角落的白色星形" in prompt
        assert "这项容错不得用于其他组织" in prompt
        assert "专用 NTD/NTDTV 字标的字母" in prompt
        assert "无法满足专属特征时省略该候选" in prompt
    else:
        assert "额外组织名单为空，禁止输出 listed_organization" in prompt
        assert '"name": "新唐人"' not in prompt
    assert "china_map_missing_region" in prompt
    assert "bbox_2d" in prompt and "0–1000" in prompt


@pytest.mark.asyncio
async def test_truncation_retries_and_becomes_incomplete_with_coverage(monkeypatch):
    calls, _ = install_detector(monkeypatch, content="[]", finish="length")
    errors = []
    assert await handlers._review_stage("flags", 1, lambda: flags.detect("img"), errors) is None
    assert len(calls) == 2
    assert errors[0]["component"] == "flags"
    assert errors[0]["error_code"] == "generation_incomplete"
    coverage = ReviewCoverage()
    coverage.add([1, 2])
    assert coverage.summarize(errors)["incomplete_samples"] == 1


@pytest.mark.asyncio
async def test_timeout_and_cancellation_drain_detector_and_release_quota(monkeypatch):
    entered, exited = [], []

    class SlowClient:
        async def post(self, *args, **kwargs):
            await asyncio.Event().wait()

    @asynccontextmanager
    async def slow_client(*args):
        entered.append(1)
        try:
            yield SlowClient()
        finally:
            exited.append(1)

    monkeypatch.setattr(flags, "model_client", slow_client)
    monkeypatch.setattr(settings, "flags_timeout_s", 0.01)
    with pytest.raises(httpx.ReadTimeout):
        await flags.detect("img")
    assert len(entered) == len(exited) == 2
    monkeypatch.setattr(settings, "flags_timeout_s", 10)
    task = asyncio.create_task(flags.detect("img"))
    while len(entered) < 3:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(exited) == 3


@pytest.mark.asyncio
async def test_flags_have_their_own_failure_circuit(monkeypatch):
    install_detector(monkeypatch, content="invalid")
    health = model_health.ModelHealth()
    token = model_health._current_health.set(health)
    try:
        for _ in range(4):
            with pytest.raises(ModelResponseError):
                await flags.detect("img")
        with pytest.raises(model_health.ModelServiceUnavailable, match="对象检测"):
            await flags.detect("img")
        assert list(health.outcomes) == ["flags"]
    finally:
        model_health._current_health.reset(token)


def quiet_siblings(monkeypatch):
    monkeypatch.setattr(settings, "flags_enabled", True)
    monkeypatch.setattr(handlers, "get_face_engine", lambda: object())
    monkeypatch.setattr(handlers, "_face_task", AsyncMock(return_value=[]))
    monkeypatch.setattr(handlers, "_call_ocr_api", AsyncMock(return_value=""))
    monkeypatch.setattr(handlers, "_review_visual", AsyncMock(return_value=None))
    guard = AsyncMock(side_effect=AssertionError("Flag discoveries must not invoke Guard"))
    monkeypatch.setattr(handlers, "_call_llm_guard", guard)
    return guard


@pytest.mark.asyncio
async def test_window_path_keeps_exact_frames_instances_and_cache_timestamps(monkeypatch):
    samples = [sample(i, value, scene=i // 3) for i, value in enumerate([0, 0, 80, 0, 160])]
    samples[0][0].timestamp = 0.0003
    samples[0][0].duration = 0.04
    samples[0][0].frame_index = 0
    install_video(monkeypatch, samples)
    quiet_siblings(monkeypatch)
    detections = flags.parse_detections("[]")
    multiple = flags.parse_detections(
        json.dumps([FLAG, {**FLAG, "bbox_2d": [100, 100, 150, 160]}, LOGO, MAP])
    )
    detector = AsyncMock(
        side_effect=[multiple, detections, ModelResponseError("flags", "invalid_json", "无法解析")]
    )
    monkeypatch.setattr(flags, "detect", detector)
    coverage = ReviewCoverage()
    progress = ReviewProgress()
    rows = await handlers._process_analyze_media(
        "https://fixture/video.mp4",
        1,
        5,
        0.5,
        include_flags=True,
        coverage=coverage,
        progress=progress,
    )
    assert detector.await_count == 3  # Exact repeated images reuse successful detections.
    found = [row for row in rows if row.get("source") == "flags"]
    assert len(found) == 12
    assert len([row for row in found if row.get("map_region") == "台湾"]) == 3
    assert {row["timestamp"] for row in found} == {"00:00:00.000", "00:00:01.000", "00:00:03.000"}
    assert all(
        "~" not in row["timestamp"] and row["review_status"] == "needs_review" for row in found
    )
    assert found[0]["object_samples"][0]["pts_seconds"] == 0.0003
    assert found[0]["object_samples"][0]["duration_seconds"] == 0.04
    assert found[0]["object_samples"][0]["frame_index"] == 0
    assert coverage.summarize(rows)["incomplete_samples"] == 1
    assert not progress.active_windows
    # Persistence consolidation cannot discard same-name instances or metadata.
    leaves = flatten_findings(consolidate_results(rows))
    assert sum(len(row.get("object_samples", [])) for row in leaves) == 12


@pytest.mark.asyncio
async def test_image_path_and_disabled_switch(monkeypatch, sample_image_bytes):
    quiet_siblings(monkeypatch)
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=sample_image_bytes))
    detector = AsyncMock(return_value=flags.parse_detections(json.dumps([FLAG, LOGO])))
    monkeypatch.setattr(flags, "detect", detector)
    rows = await handlers._process_analyze_media(
        "https://fixture/img.jpg", 1, 5, 0.5, include_flags=True
    )
    assert len(rows) == 2 and all(row["source"] == "flags" for row in rows)
    assert all(row["object_samples"][0]["time_ms"] == 0 for row in rows)
    assert await handlers._process_analyze_media("https://fixture/img.jpg", 1, 5, 0.5) == []
    assert detector.await_count == 1
    monkeypatch.setattr(settings, "flags_enabled", False)
    assert (
        await handlers._process_analyze_media(
            "https://fixture/img.jpg", 1, 5, 0.5, include_flags=True
        )
        == []
    )
    assert detector.await_count == 1


@pytest.mark.asyncio
async def test_target_path_preserves_pts_and_fixed_sample_selection(monkeypatch):
    quiet_siblings(monkeypatch)
    samples = [sample(i, i) for i in range(3)]
    samples[0][0].timestamp = 0.0003
    samples[0][0].duration = 0.04
    samples[0][0].frame_index = 0
    samples[1].sampled = False

    class Sampler:
        interval = 1

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter(samples)

    monkeypatch.setattr(settings, "nsfw_review_mode", "target")
    monkeypatch.setattr(handlers, "VideoFrameSampler", Sampler)
    monkeypatch.setattr(handlers, "_download_video_safe_async", AsyncMock())
    detector = AsyncMock(return_value=flags.parse_detections(json.dumps([FLAG])))
    monkeypatch.setattr(flags, "detect", detector)
    rows = await handlers._process_analyze_media(
        "https://fixture/video.mp4", 1, 5, 0.5, include_flags=True
    )
    assert detector.await_count == 2
    assert [row["object_samples"][0]["pts_seconds"] for row in rows] == [0.0003, 2]
    assert rows[0]["object_samples"][0]["duration_seconds"] == 0.04


@pytest.mark.asyncio
async def test_dedicated_visual_path_does_not_run_flags(monkeypatch):
    install_video(monkeypatch, [sample(0)])
    quiet_siblings(monkeypatch)
    detector = AsyncMock(side_effect=AssertionError("Flags should be opt-in per endpoint"))
    monkeypatch.setattr(flags, "detect", detector)
    await review_windows.analyze_video("https://fixture/video.mp4", 1)
    detector.assert_not_awaited()


def test_flag_configuration_is_disabled_by_default_and_validated():
    assert Settings(_env_file=None).flags_enabled is False
    assert Settings(_env_file=None).flags_reasoning_effort == "none"
    with pytest.raises(ValueError):
        Settings(_env_file=None, flags_max_tokens=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, flags_reasoning_effort="unsupported")


def test_flag_parameters_remain_frozen_for_a_running_review(monkeypatch):
    from wcm_facerec import runtime_parameters

    previous = runtime_parameters.snapshot()
    try:
        runtime_parameters.install(
            {"flags_enabled": True, "flags_model": "first", "flags_max_tokens": 2048}
        )
        frozen = runtime_parameters.snapshot()
        with runtime_parameters.frozen(frozen):
            runtime_parameters.install(
                {"flags_enabled": False, "flags_model": "second", "flags_max_tokens": 4096}
            )
            assert settings.flags_enabled and settings.flags_model == "first"
            assert settings.flags_max_tokens == 2048
        assert not settings.flags_enabled and settings.flags_model == "second"
    finally:
        runtime_parameters.install(previous)


@pytest.mark.parametrize("target", ["exposed_genitals", "exposed_female_nipple", "exposed_anus"])
def test_nudity_uses_anatomical_names_review_category_and_point_boxes(target):
    from wcm_facerec.object_detection_policy import NUDITY_LABELS

    detections = flags.parse_detections(json.dumps([{**NUDITY, "target": target}]))
    rows = flags.findings(detections, "00:00:01.000", 1)
    assert rows[0]["category"] == "裸露部位"
    assert rows[0]["name"] == NUDITY_LABELS[target]
    assert rows[0]["object_type"] == "nudity"
    assert rows[0]["object_target"] == target
    assert rows[0]["object_evidence"] == NUDITY["evidence"]
    assert rows[0]["review_status"] == "needs_review"
    assert rows[0]["object_samples"][0]["bbox"] == {"x": 0.1, "y": 0.2, "w": 0.06, "h": 0.06}
    coverage = ReviewCoverage()
    coverage.add([1])
    assert coverage.summarize(rows)["incomplete_checks"] == 0


@pytest.mark.asyncio
async def test_positive_negative_prompts_and_organization_names_are_frozen_and_used(monkeypatch):
    from wcm_facerec import runtime_parameters

    previous = runtime_parameters.snapshot()
    try:
        frozen = {
            "flags_positive_prompt": "原正向",
            "flags_negative_prompt": "原反向",
            "flags_organization_targets": ["新唐人"],
        }
        calls, _ = install_detector(monkeypatch)
        with runtime_parameters.frozen(frozen):
            runtime_parameters.install(
                {
                    "flags_positive_prompt": "新正向",
                    "flags_negative_prompt": "新反向",
                    "flags_organization_targets": [],
                }
            )
            assert len(await flags.detect("img")) == 2
            prompt = calls[0]["messages"][0]["content"][0]["text"]
            assert "原正向" in prompt and "原反向" in prompt
            assert "NTD" in prompt and "新唐人" in prompt
            assert "新正向" not in prompt and "新中国联邦" not in prompt
        with pytest.raises(ModelResponseError, match="不在指定名单"):
            flags.parse_detections(json.dumps([LOGO]))
    finally:
        runtime_parameters.install(previous)


@pytest.mark.asyncio
async def test_mixed_symbol_nudity_and_map_findings_survive_full_image_review(
    monkeypatch, sample_image_bytes
):
    quiet_siblings(monkeypatch)
    monkeypatch.setattr(handlers, "_download_url_safe", AsyncMock(return_value=sample_image_bytes))
    install_detector(monkeypatch, content=json.dumps([FLAG, LOGO, NUDITY, MAP]))
    rows = await handlers._process_analyze_media(
        "https://fixture/img.jpg", 1, 5, 0.5, include_flags=True
    )
    leaves = list(flatten_findings(consolidate_results(rows)))
    assert len(leaves) == 4
    assert {row["object_type"] for row in leaves} == {"flag", "logo", "nudity", "map"}
    assert {row["category"] for row in leaves} == {"旗帜与徽标", "裸露部位", "地图核查"}
    assert all(row["object_target"] and row["object_evidence"] for row in leaves)


@pytest.mark.parametrize("region", ["台湾", "香港", "澳门", "西藏", "新疆", "藏南", "阿克赛钦"])
@pytest.mark.parametrize(
    "target", ["china_map_missing_region", "china_map_separate_region", "china_map_foreign_region"]
)
def test_map_regions_become_review_candidates_with_exact_point_evidence(region, target):
    detections = flags.parse_detections(
        json.dumps([{**MAP, "map_region": region, "target": target}])
    )
    frame = sample(1, 0)[0]
    frame.timestamp, frame.duration, frame.frame_index = 1.0003, 0.04, 25
    rows = flags.findings(detections, "00:00:01.000", frame.timestamp, frame=frame)
    restored = list(flatten_findings(consolidate_results(rows)))
    assert restored[0]["category"] == "地图核查"
    assert restored[0]["map_region"] == region
    assert restored[0]["object_target"] == target
    assert region in restored[0]["description"]
    assert restored[0]["object_samples"][0] == {
        "time_ms": 1000,
        "pts_seconds": 1.0003,
        "duration_seconds": 0.04,
        "frame_index": 25,
        "bbox": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
    }
    coverage = ReviewCoverage()
    coverage.add([1])
    assert coverage.summarize(rows)["incomplete_checks"] == 0


@pytest.mark.parametrize(
    "region,old_name",
    [
        ("符拉迪沃斯托克", "海参崴"),
        ("布拉戈维申斯克", "海兰泡"),
        ("乌苏里斯克", "双城子"),
        ("哈巴罗夫斯克", "伯力"),
        ("萨哈林岛", "库页岛"),
        ("涅尔琴斯克", "尼布楚"),
        ("尼古拉耶夫斯克", "庙街"),
        ("斯塔诺夫山脉", "外兴安岭"),
    ],
)
def test_map_name_pairs_are_canonical_and_already_present_names_are_excluded(region, old_name):
    item = {**MAP, "target": "map_missing_chinese_name", "map_region": region, "map_text": region}
    parsed = flags.parse_detections(json.dumps([item]))
    rows = flags.findings(parsed, "00:00:00.000", 0)
    assert rows[0]["map_reference_name"] == old_name
    assert rows[0]["map_text"] == region
    assert old_name in rows[0]["description"]
    assert (
        flags.parse_detections(json.dumps([{**item, "map_text": f"{region}（{old_name}）"}])) == []
    )


def test_map_traditional_name_with_spacing_is_not_missing():
    item = {
        **MAP,
        "target": "map_missing_chinese_name",
        "map_region": "符拉迪沃斯托克",
        "map_text": "符拉迪沃斯托克\n（海 參 崴）",
    }
    assert flags.parse_detections(json.dumps([item])) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"map_region": "日本"},
        {"map_region": ["台湾"]},
        {"map_region": None},
        {"category": "flag"},
        {"target": "tibet_related"},
        {"map_text": 123},
        {"target": "map_missing_chinese_name", "map_region": "台湾", "map_text": "台湾"},
        {"target": "map_missing_chinese_name", "map_region": "莫斯科", "map_text": "莫斯科"},
        {"target": "map_missing_chinese_name", "map_region": "符拉迪沃斯托克"},
    ],
)
def test_invalid_map_scope_or_evidence_is_a_stage_error(overrides):
    with pytest.raises(ModelResponseError):
        flags.parse_detections(json.dumps([{**MAP, **overrides}]))


def test_map_prompt_additions_do_not_require_replacing_existing_prompt_parameters():
    from wcm_facerec import runtime_parameters

    with runtime_parameters.frozen(
        {"flags_positive_prompt": "原正向", "flags_negative_prompt": "原反向"}
    ):
        prompt = flags.build_prompt()
        assert "原正向" in prompt and "原反向" in prompt
        assert "china_map_missing_region" in prompt and "map_missing_chinese_name" in prompt
        assert "图外区域不做缺失推断" in prompt and "海兰泡" in prompt
    with runtime_parameters.frozen(
        {
            "flags_map_positive_prompt": "自定义地图正向",
            "flags_map_negative_prompt": "自定义地图反向",
        }
    ):
        prompt = flags.build_prompt()
        assert "自定义地图正向" in prompt and "自定义地图反向" in prompt
