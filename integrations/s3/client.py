"""Async S3 client wrapping boto3.

In development (ENVIRONMENT=development), endpoint_url defaults to MinIO at
http://localhost:9000 unless S3_ENDPOINT_URL is set explicitly.
In production, endpoint_url=None → standard AWS S3.
"""

import asyncio
import os
import uuid

import boto3
import structlog
from botocore.config import Config
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_RETRY = dict(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True
)


class S3Client:
    def __init__(
        self,
        *,
        bucket: str,
        region: str,
        access_key: str,
        secret_key: str,
        endpoint_url: str | None = None,
    ):
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    @classmethod
    def from_settings(cls, settings) -> "S3Client":
        endpoint = settings.S3_ENDPOINT_URL
        if endpoint is None and settings.ENVIRONMENT == "development":
            endpoint = "http://localhost:9000"
        return cls(
            bucket=settings.S3_BUCKET_NAME,
            region=settings.AWS_REGION,
            access_key=settings.AWS_ACCESS_KEY_ID,
            secret_key=settings.AWS_SECRET_ACCESS_KEY,
            endpoint_url=endpoint,
        )

    @staticmethod
    def _make_key(filename: str) -> str:
        return f"uploads/{uuid.uuid4()}_{filename}"

    @retry(**_RETRY)
    def _upload_sync(self, file_obj, key: str) -> None:
        self._client.upload_fileobj(file_obj, self._bucket, key)

    async def upload_fileobj(self, file_obj, filename: str) -> str:
        key = self._make_key(filename)
        await asyncio.to_thread(self._upload_sync, file_obj, key)
        return key

    @retry(**_RETRY)
    def _download_sync(self, key: str) -> str:
        tmp_dir = "tmp_uploads"
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = os.path.join(tmp_dir, key.replace("/", "_"))
        with open(local_path, "wb") as f:
            self._client.download_fileobj(self._bucket, key, f)
        return local_path

    async def download_to_temp(self, key: str) -> str:
        return await asyncio.to_thread(self._download_sync, key)

    @retry(**_RETRY)
    def _delete_sync(self, keys: list[str]) -> None:
        if not keys:
            return
        self._client.delete_objects(
            Bucket=self._bucket,
            Delete={"Objects": [{"Key": k} for k in keys]},
        )

    async def delete_objects(self, keys: list[str]) -> None:
        await asyncio.to_thread(self._delete_sync, keys)
