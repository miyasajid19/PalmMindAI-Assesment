from functools import lru_cache

import boto3
from botocore.config import Config

from app.core.config import settings


def _get_required_setting(name: str, value: str | None) -> str:
    if not value or not value.strip():
        raise ValueError(f"Missing required B2 configuration: {name}")

    return value.strip()


@lru_cache(maxsize=1)
def get_b2_client():
    """
    Create and cache the Backblaze B2 S3-compatible client.

    The client is initialized lazily, so importing this module
    does not require B2 credentials to be present immediately.
    """

    endpoint_url = _get_required_setting(
        "B2_ENDPOINT_URL",
        settings.B2_ENDPOINT_URL,
    )

    region = _get_required_setting(
        "B2_REGION",
        settings.B2_REGION,
    )

    key_id = _get_required_setting(
        "B2_KEY_ID",
        settings.B2_KEY_ID,
    )

    application_key = _get_required_setting(
        "B2_APPLICATION_KEY",
        settings.B2_APPLICATION_KEY,
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url.rstrip("/"),
        aws_access_key_id=key_id,
        aws_secret_access_key=application_key,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=30,
            read_timeout=60,
            retries={
                "max_attempts": 3,
                "mode": "standard",
            },
        ),
    )


def get_b2_bucket_name() -> str:
    return _get_required_setting(
        "B2_BUCKET_NAME",
        settings.B2_BUCKET_NAME,
    )


def get_b2_storage_prefix() -> str:
    prefix = settings.B2_STORAGE_PREFIX or "voice-ai/"
    return prefix.strip("/") + "/"