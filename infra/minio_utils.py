"""Centralized MinIO helper utilities for I/O operations."""
from __future__ import annotations

import io
import json
from typing import Any, Iterable

from infra.resources import MinIOResource
from core.settings import SETTINGS


async def get_minio() -> MinIOResource:
    """Initialize and return a MinIOResource."""
    minio = MinIOResource(
        endpoint=SETTINGS.MINIO.MINIO_ENDPOINT,
        access_key=SETTINGS.MINIO.MINIO_ACCESS_KEY,
        secret_key=SETTINGS.MINIO.MINIO_SECRET_KEY.get_secret_value(),
        bucket_name=SETTINGS.MINIO.MINIO_BUCKET,
    )
    await minio.init()
    return minio


def put_jsonl(
    minio: MinIOResource, bucket: str, key: str, items: Iterable[dict]
) -> None:
    payload = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items).encode(
        "utf-8"
    )
    client = minio.client
    client.put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/jsonl",
    )


def put_json(minio: MinIOResource, bucket: str, key: str, obj: Any) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    client = minio.client
    client.put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )


def put_text(
    minio: MinIOResource,
    bucket: str,
    key: str,
    text: str,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
    payload = text.encode("utf-8")
    client = minio.client
    client.put_object(
        bucket,
        key,
        io.BytesIO(payload),
        length=len(payload),
        content_type=content_type,
    )


def read_object_bytes(minio: MinIOResource, bucket: str, key: str) -> bytes:
    client = minio.client
    resp = client.get_object(bucket, key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()
