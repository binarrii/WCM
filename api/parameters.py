"""General application parameter management API."""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

from . import parameter_store

parameters_bp = APIRouter()
_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


class ParameterValue(BaseModel):
    value: Any
    type: Literal["string", "number", "boolean", "enum", "json"]
    group: str = Field(min_length=1, max_length=100)
    options: list[Any] | None = None

    @field_validator("group")
    @classmethod
    def clean_group(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("参数分组不能为空")
        return cleaned

    @model_validator(mode="after")
    def validate_typed_value(self):
        parameter_store.encode_value(self.value, self.type, self.options)
        return self


class ParameterCreate(ParameterValue):
    key: str = Field(min_length=1, max_length=191)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        cleaned = value.strip()
        if not _KEY_PATTERN.fullmatch(cleaned):
            raise ValueError("key 必须以字母开头，且只能包含字母、数字、点、横线和下划线")
        return cleaned


def _storage_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


@parameters_bp.get("/parameters")
async def list_parameters():
    return parameter_store.list_parameters()


@parameters_bp.post("/parameters", status_code=201)
async def create_parameter(body: ParameterCreate):
    try:
        return await parameter_store.create(
            body.key, body.value, body.type, body.group, body.options
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except parameter_store.ParameterAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except parameter_store.ParameterStoreUnavailable as exc:
        raise _storage_error(exc) from exc


@parameters_bp.put("/parameters/{key}")
async def update_parameter(key: str, body: ParameterValue):
    if not _KEY_PATTERN.fullmatch(key):
        raise HTTPException(status_code=422, detail="无效的参数 key")
    try:
        return await parameter_store.update(key, body.value, body.type, body.group, body.options)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except parameter_store.ParameterNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except parameter_store.ParameterStoreUnavailable as exc:
        raise _storage_error(exc) from exc


@parameters_bp.delete("/parameters/{key}")
async def delete_parameter(key: str):
    if not _KEY_PATTERN.fullmatch(key):
        raise HTTPException(status_code=422, detail="无效的参数 key")
    try:
        await parameter_store.delete(key)
    except parameter_store.ParameterProtected as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except parameter_store.ParameterNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except parameter_store.ParameterStoreUnavailable as exc:
        raise _storage_error(exc) from exc
    return {"deleted": 1, "key": key}
