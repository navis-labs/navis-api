"""Object storage (S3) — attachments, exports, raw signal payloads.

Lazily constructed so the app boots without AWS credentials in dev. Credentials
come from settings if present, otherwise boto3's default chain (IAM role, env).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from navis.platform.config import settings


@lru_cache
def get_s3_client() -> Any:
    import boto3  # imported lazily; boto3 is heavy

    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    return boto3.client("s3", **kwargs)
