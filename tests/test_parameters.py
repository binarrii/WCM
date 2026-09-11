from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from api import parameter_store, parameters
from api.main import create_app
from wcm_facerec import runtime_parameters
from wcm_facerec.config import BUSINESS_PARAMETER_SPECS, settings

EXPECTED_BUSINESS_PARAMETERS = {
    "insightface_base_url",
    "insightface_model_name",
    "insightface_collection_id",
    "insightface_api_key",
    "insightface_timeout_s",
    "insightface_verify_similarity_threshold",
    "insightface_quality_weight",
    "insightface_adaptive_threshold_step",
    "insightface_norm_reference",
    "insightface_category_collections",
    "default_category",
    "face_profile_optimization",
    "face_crop_padding",
    "face_candidate_similarity",
    "face_high_similarity",
    "face_min_candidate_margin",
    "face_min_confirming_frames",
    "face_profile_pose_threshold",
    "face_low_sharpness_threshold",
    "face_track_max_gap_s",
    "face_neighbor_offsets_s",
    "face_max_extra_frames_per_window",
    "face_max_extra_call_ratio",
    "face_neighbor_concurrency",
    "face_gallery_target_samples",
    "max_file_size_mb",
    "review_task_concurrency",
    "review_window_concurrency",
    "jpeg_quality",
    "visual_timeout_s",
    "ocr_timeout_s",
    "guard_timeout_s",
    "model_api_url",
    "model_api_key",
    "nsfw_image_mode",
    "nsfw_verify_target",
    "nsfw_sampling_mode",
    "nsfw_scene_max_stride",
    "nsfw_scene_cut_threshold",
    "nsfw_review_mode",
    "nsfw_window_max_seconds",
}


def _row(key, value, value_type="string", group="default"):
    return {
        "config_key": key,
        "config_value": value,
        "value_type": value_type,
        "group_name": group,
        "created_at": None,
        "updated_at": None,
    }


@pytest.mark.parametrize(
    ("value", "value_type", "encoded"),
    [
        ("hello", "string", "hello"),
        (12.5, "number", "12.5"),
        ({"enabled": True, "tags": ["a", "b"]}, "json", '{"enabled":true,"tags":["a","b"]}'),
    ],
)
def test_typed_parameter_values_round_trip(value, value_type, encoded):
    assert parameter_store.encode_value(value, value_type) == encoded
    assert parameter_store.decode_value(encoded, value_type) == value


@pytest.mark.parametrize(
    ("value", "value_type"),
    [(1, "string"), (True, "number"), ("12", "number"), (float("inf"), "number"), (set(), "json")],
)
def test_typed_parameter_values_reject_mismatches(value, value_type):
    with pytest.raises(ValueError):
        parameter_store.encode_value(value, value_type)


def test_memory_snapshot_is_defensive_and_visible_to_existing_threads():
    parameter_store._install_snapshot([_row("feature.flags", '{"enabled":false}', "json")])
    first_read_complete = Event()
    read_again = Event()

    def read_on_worker_thread():
        first = parameter_store.get("feature.flags")
        first["mutated_by_reader"] = True
        first_read_complete.set()
        assert read_again.wait(timeout=2)
        return parameter_store.get("feature.flags")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(read_on_worker_thread)
            assert first_read_complete.wait(timeout=2)
            assert parameter_store.get("feature.flags") == {"enabled": False}
            parameter_store._install_snapshot([_row("feature.flags", '{"enabled":true}', "json")])
            read_again.set()
            assert result.result(timeout=2) == {"enabled": True}
    finally:
        parameter_store._install_snapshot([])


def test_all_business_settings_are_declared_and_bootstrap_settings_are_excluded():
    assert set(BUSINESS_PARAMETER_SPECS) == EXPECTED_BUSINESS_PARAMETERS
    assert not EXPECTED_BUSINESS_PARAMETERS.intersection(
        {
            "api_host",
            "api_port",
            "review_tasks_db_enabled",
            "review_tasks_db_host",
            "review_tasks_db_port",
            "review_tasks_db_name",
            "review_tasks_db_user",
            "review_tasks_db_password",
            "review_tasks_db_connect_timeout_s",
            "data_root",
        }
    )


def test_builtin_snapshot_overlays_settings_and_defensively_copies_json():
    spec = BUSINESS_PARAMETER_SPECS["insightface_category_collections"]
    parameter_store._install_snapshot(
        [_row("insightface_category_collections", '{"测试":"test"}', "json", spec.group)]
    )
    try:
        assert settings.insightface_category_collections == {"测试": "test"}
        value = settings.insightface_category_collections
        value["mutated"] = "locally"
        assert settings.insightface_category_collections == {"测试": "test"}
        assert runtime_parameters.snapshot() == {
            "insightface_category_collections": {"测试": "test"}
        }
    finally:
        parameter_store._install_snapshot([])


def test_secret_parameters_are_masked_but_remain_available_to_server_code():
    spec = BUSINESS_PARAMETER_SPECS["model_api_key"]
    parameter_store._install_snapshot([_row("model_api_key", "top-secret", "string", spec.group)])
    try:
        item = parameter_store.list_parameters()["items"][0]
        assert item["value"] is None
        assert item["built_in"] is True
        assert item["secret"] is True
        assert item["has_value"] is True
        assert parameter_store.get("model_api_key") == "top-secret"
        assert settings.model_api_key == "top-secret"
    finally:
        parameter_store._install_snapshot([])


@pytest.mark.parametrize(
    ("key", "value", "value_type", "group"),
    [
        ("jpeg_quality", 101, "number", "审核调度"),
        ("nsfw_image_mode", "grid", "string", "内容审核"),
        ("face_profile_optimization", True, "string", "人脸优化"),
        ("model_api_url", "https://example.com", "string", "错误分组"),
    ],
)
def test_builtin_parameter_type_group_and_domain_validation(key, value, value_type, group):
    with pytest.raises(ValueError):
        parameter_store._encoded_value(key, value, value_type, group)


def test_builtin_rows_with_manually_changed_groups_are_rejected():
    with pytest.raises(parameter_store.ParameterStoreUnavailable, match="分组必须是"):
        parameter_store._cached_parameter(
            _row("model_api_url", "https://example.com", "string", "错误分组")
        )


def test_business_environment_variables_are_removed_from_runtime_manifests():
    root = Path(__file__).resolve().parents[1]
    compose_manifest = (root / "compose.yaml").read_text()
    runtime_manifests = [
        compose_manifest,
        (root / ".env.example").read_text(),
        (root / "Dockerfile").read_text(),
    ]
    for key in EXPECTED_BUSINESS_PARAMETERS:
        env_name = f"WCM_{key.upper()}"
        assert all(env_name not in manifest for manifest in runtime_manifests)
    for env_name in (
        "WCM_API_HOST",
        "WCM_API_PORT",
        "WCM_REVIEW_TASKS_DB_HOST",
        "WCM_REVIEW_TASKS_DB_PORT",
    ):
        assert env_name in compose_manifest


def test_initialization_seeds_every_builtin_without_overwriting_existing_rows(monkeypatch):
    statements = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement, params=None):
            statements.append((statement, params))
            return 1

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(parameter_store, "_connect", lambda: FakeConnection())
    parameter_store._initialize_sync()

    inserts = [
        params
        for statement, params in statements
        if "INSERT IGNORE INTO system_parameters" in statement
    ]
    assert len(inserts) == len(EXPECTED_BUSINESS_PARAMETERS)
    assert {params[0] for params in inserts} == EXPECTED_BUSINESS_PARAMETERS
    assert all(params[2] == BUSINESS_PARAMETER_SPECS[params[0]].value_type for params in inserts)
    assert all(params[3] == BUSINESS_PARAMETER_SPECS[params[0]].group for params in inserts)


def test_parameter_crud_routes(monkeypatch):
    item = {
        "key": "review.max_retries",
        "value": 3,
        "type": "number",
        "group": "review",
        "created_at": None,
        "updated_at": None,
    }
    list_parameters = MagicMock(
        return_value={"items": [item], "total": 1, "loaded_at": None, "version": 1}
    )
    create = AsyncMock(return_value=item)
    update = AsyncMock(return_value={**item, "value": 4})
    delete = AsyncMock()
    monkeypatch.setattr(parameters.parameter_store, "list_parameters", list_parameters)
    monkeypatch.setattr(parameters.parameter_store, "create", create)
    monkeypatch.setattr(parameters.parameter_store, "update", update)
    monkeypatch.setattr(parameters.parameter_store, "delete", delete)

    client = TestClient(create_app())
    listed = client.get("/api/v1/parameters")
    created = client.post(
        "/api/v1/parameters", json={**item, "created_at": None, "updated_at": None}
    )
    updated = client.put(
        "/api/v1/parameters/review.max_retries",
        json={"value": 4, "type": "number", "group": "review"},
    )
    deleted = client.delete("/api/v1/parameters/review.max_retries")

    assert listed.json()["items"][0]["key"] == "review.max_retries"
    assert created.status_code == 201
    assert updated.json()["value"] == 4
    assert deleted.json() == {"deleted": 1, "key": "review.max_retries"}
    create.assert_awaited_once_with("review.max_retries", 3, "number", "review")
    update.assert_awaited_once_with("review.max_retries", 4, "number", "review")
    delete.assert_awaited_once_with("review.max_retries")


@pytest.mark.parametrize(
    "payload",
    [
        {"key": "bad key", "value": "x", "type": "string", "group": "default"},
        {"key": "valid.key", "value": "3", "type": "number", "group": "default"},
        {"key": "valid.key", "value": 3, "type": "string", "group": "default"},
        {"key": "valid.key", "value": {}, "type": "unknown", "group": "default"},
        {"key": "valid.key", "value": {}, "type": "json", "group": "  "},
    ],
)
def test_parameter_route_validates_keys_groups_and_typed_values(monkeypatch, payload):
    monkeypatch.setattr(parameters.parameter_store, "create", AsyncMock())
    response = TestClient(create_app()).post("/api/v1/parameters", json=payload)
    assert response.status_code == 422


def test_parameter_route_maps_storage_conflicts(monkeypatch):
    monkeypatch.setattr(
        parameters.parameter_store,
        "create",
        AsyncMock(side_effect=parameter_store.ParameterAlreadyExists("参数 duplicate 已存在")),
    )
    response = TestClient(create_app()).post(
        "/api/v1/parameters",
        json={"key": "duplicate", "value": "x", "type": "string", "group": "default"},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "参数 duplicate 已存在"


def test_parameter_route_protects_builtin_parameters_from_deletion(monkeypatch):
    monkeypatch.setattr(
        parameters.parameter_store,
        "delete",
        AsyncMock(side_effect=parameter_store.ParameterProtected("内置参数不能删除")),
    )
    response = TestClient(create_app()).delete("/api/v1/parameters/jpeg_quality")
    assert response.status_code == 409
    assert response.json()["detail"] == "内置参数不能删除"
