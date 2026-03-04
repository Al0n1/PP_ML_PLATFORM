"""
Service for working with Yandex Object Storage via the S3-compatible API.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Optional

import aioboto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from src.config.services.ya_s3_config import settings
from src.utils.sse_messages import build_error, build_progress, build_success

logger = logging.getLogger(__name__)

STAGE_PROGRESS = {
    "initializing": 5,
    "uploading": 70,
    "downloading": 70,
    "deleting": 50,
    "listing": 50,
}


class YaS3Service:
    """
    Service boundary for Yandex Object Storage.
    """

    S3_ERROR_CODES = {
        "NoSuchBucket": "Bucket does not exist",
        "NoSuchKey": "Object not found",
        "AccessDenied": "Access denied",
        "InvalidAccessKeyId": "Invalid access key id",
        "SignatureDoesNotMatch": "Invalid secret access key",
        "BucketAlreadyExists": "Bucket already exists",
        "BucketNotEmpty": "Bucket is not empty",
        "EntityTooLarge": "File is too large",
        "InvalidBucketName": "Invalid bucket name",
        "KeyTooLong": "Object key is too long",
        "ServiceUnavailable": "Service temporarily unavailable",
        "RequestTimeout": "Request timeout",
    }

    def __init__(self):
        self._session: Optional[aioboto3.Session] = None
        self._boto_config = BotoConfig(
            region_name=settings.YA_S3_REGION_NAME,
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},
            retries={
                "max_attempts": settings.YA_S3_MAX_RETRIES,
                "mode": "adaptive",
            },
            connect_timeout=30,
            read_timeout=settings.YA_S3_OPERATION_TIMEOUT_SECONDS,
        )
        logger.info("YaS3Service initialized")

    def _get_session(self) -> aioboto3.Session:
        if self._session is None:
            self._session = aioboto3.Session(
                aws_access_key_id=settings.YA_S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.YA_S3_SECRET_ACCESS_KEY,
                region_name=settings.YA_S3_REGION_NAME,
            )
        return self._session

    def _get_s3_client(self):
        session = self._get_session()
        return session.client(
            "s3",
            endpoint_url=settings.YA_S3_ENDPOINT_URL,
            config=self._boto_config,
        )

    @staticmethod
    def _success(result: dict | None = None) -> dict:
        return build_success(result=result)

    @staticmethod
    def _error(
        code: str,
        message: str,
        stage_failed: str,
        details: str | None = None,
        recoverable: bool = True,
    ) -> dict:
        return build_error(
            code=code,
            message=message,
            stage_failed=stage_failed,
            details=details,
            recoverable=recoverable,
        )

    def _progress(
        self,
        stage: str,
        *,
        progress: int | None = None,
        details: dict | None = None,
    ) -> dict:
        return build_progress(
            progress=progress if progress is not None else STAGE_PROGRESS[stage],
            stage=stage,
            details=details,
        )

    def _calculate_md5(self, file_path: Path) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _handle_s3_error(self, error: Exception, operation: str) -> str:
        if isinstance(error, ClientError):
            error_code = error.response.get("Error", {}).get("Code", "Unknown")
            error_message = self.S3_ERROR_CODES.get(error_code, f"S3 error: {error_code}")
            logger.error("S3 ClientError during %s: %s - %s", operation, error_code, error_message)
            return f"{operation}: {error_message}"
        if isinstance(error, BotoCoreError):
            logger.error("BotoCoreError during %s: %s", operation, error)
            return f"{operation}: S3 connection error"
        logger.error("Unexpected error during %s: %s", operation, error)
        return f"{operation}: {error}"

    async def execute_stream(self, data: dict) -> AsyncIterator[dict]:
        operation_data = data.get("data", {})
        operation = operation_data.get("operation", "").lower()

        logger.info("YaS3Service.execute_stream: operation=%s, data=%s", operation, operation_data)

        if not operation:
            yield self._error(
                code="OPERATION_MISSING",
                message="Operation is required",
                stage_failed="validation",
            )
            return

        if operation not in {"upload", "download", "delete", "list"}:
            yield self._error(
                code="OPERATION_UNKNOWN",
                message=f"Unknown operation: {operation}",
                stage_failed="validation",
            )
            return

        try:
            if operation == "upload":
                async for message in self._execute_upload_stream(operation_data):
                    yield message
            elif operation == "download":
                async for message in self._execute_download_stream(operation_data):
                    yield message
            elif operation == "delete":
                async for message in self._execute_delete_stream(operation_data):
                    yield message
            else:
                async for message in self._execute_list_stream(operation_data):
                    yield message
        except Exception as exc:
            error_message = self._handle_s3_error(exc, operation)
            logger.exception("Error in execute_stream for operation %s", operation)
            yield self._error(
                code="SERVICE_ERROR",
                message=error_message,
                stage_failed=operation,
                details=str(exc),
            )

    async def _execute_upload_stream(self, data: dict) -> AsyncIterator[dict]:
        file_path_str = data.get("file_path")
        object_key = data.get("object_key")

        if not file_path_str:
            yield self._error(
                code="FILE_PATH_MISSING",
                message="file_path is required",
                stage_failed="validation",
            )
            return

        file_path = Path(file_path_str)
        if not file_path.exists():
            yield self._error(
                code="FILE_NOT_FOUND",
                message=f"File not found: {file_path}",
                stage_failed="validation",
            )
            return

        if not object_key:
            object_key = file_path.name

        file_size = file_path.stat().st_size

        yield self._progress("initializing")
        logger.info("Starting upload: %s -> s3://%s/%s", file_path, settings.YA_S3_BUCKET_NAME, object_key)

        try:
            md5_hash = self._calculate_md5(file_path)

            if file_size > settings.multipart_threshold_bytes:
                async for message in self._upload_file_multipart(file_path, object_key, file_size):
                    yield message
            else:
                async for message in self._upload_file_simple(file_path, object_key):
                    yield message

            presigned_url = await self.generate_presigned_url(object_key)
            public_url = settings.get_public_url(object_key)

            result = {
                "object_key": object_key,
                "bucket": settings.YA_S3_BUCKET_NAME,
                "size": file_size,
                "md5": md5_hash,
                "public_url": public_url,
                "download_url": presigned_url,
                "url_expires_in_hours": settings.YA_S3_SIGNED_URL_EXPIRATION_HOURS,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }

            yield self._success(result=result)
        except Exception as exc:
            error_message = self._handle_s3_error(exc, "upload")
            logger.exception("Upload failed for %s", file_path)
            yield self._error(
                code="UPLOAD_FAILED",
                message=error_message,
                stage_failed="uploading",
                details=str(exc),
            )

    async def _upload_file_simple(self, file_path: Path, object_key: str) -> AsyncIterator[dict]:
        async with self._get_s3_client() as s3_client:
            with open(file_path, "rb") as file_obj:
                await s3_client.put_object(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Key=object_key,
                    Body=file_obj,
                )

        yield self._progress("uploading", progress=95)

    async def _upload_file_multipart(self, file_path: Path, object_key: str, file_size: int) -> AsyncIterator[dict]:
        chunk_size = settings.multipart_chunk_size_bytes
        total_parts = (file_size + chunk_size - 1) // chunk_size

        async with self._get_s3_client() as s3_client:
            response = await s3_client.create_multipart_upload(
                Bucket=settings.YA_S3_BUCKET_NAME,
                Key=object_key,
            )
            upload_id = response["UploadId"]
            logger.info("Multipart upload initiated: upload_id=%s, total_parts=%s", upload_id, total_parts)

            try:
                parts = []
                with open(file_path, "rb") as file_obj:
                    for part_number in range(1, total_parts + 1):
                        chunk = file_obj.read(chunk_size)
                        part_response = await s3_client.upload_part(
                            Bucket=settings.YA_S3_BUCKET_NAME,
                            Key=object_key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk,
                        )
                        parts.append(
                            {
                                "PartNumber": part_number,
                                "ETag": part_response["ETag"],
                            }
                        )

                        progress = min(95, 70 + int((part_number / total_parts) * 25))
                        yield self._progress(
                            "uploading",
                            progress=progress,
                            details={
                                "current_step": part_number,
                                "total_steps": total_parts,
                            },
                        )

                await s3_client.complete_multipart_upload(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Key=object_key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )
                logger.info("Multipart upload completed: %s", object_key)
            except Exception:
                logger.exception("Multipart upload failed, aborting")
                try:
                    await s3_client.abort_multipart_upload(
                        Bucket=settings.YA_S3_BUCKET_NAME,
                        Key=object_key,
                        UploadId=upload_id,
                    )
                except Exception:
                    logger.exception("Failed to abort multipart upload")
                raise

    async def _execute_download_stream(self, data: dict) -> AsyncIterator[dict]:
        object_key = data.get("object_key")
        download_path_str = data.get("download_path")

        if not object_key:
            yield self._error(
                code="OBJECT_KEY_MISSING",
                message="object_key is required",
                stage_failed="validation",
            )
            return

        if download_path_str:
            download_path = Path(download_path_str)
        else:
            temp_dir = Path(os.getenv("TEMP_DIR", "var/temp"))
            temp_dir.mkdir(parents=True, exist_ok=True)
            download_path = temp_dir / Path(object_key).name

        yield self._progress("initializing")
        logger.info("Starting download: s3://%s/%s -> %s", settings.YA_S3_BUCKET_NAME, object_key, download_path)

        try:
            async with self._get_s3_client() as s3_client:
                head_response = await s3_client.head_object(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Key=object_key,
                )
                file_size = head_response["ContentLength"]

                download_path.parent.mkdir(parents=True, exist_ok=True)
                yield self._progress("downloading")

                await s3_client.download_file(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Key=object_key,
                    Filename=str(download_path),
                )

            result = {
                "object_key": object_key,
                "download_path": str(download_path),
                "size": file_size,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            }
            yield self._success(result=result)
        except Exception as exc:
            error_message = self._handle_s3_error(exc, "download")
            logger.exception("Download failed for %s", object_key)
            yield self._error(
                code="DOWNLOAD_FAILED",
                message=error_message,
                stage_failed="downloading",
                details=str(exc),
            )

    async def _execute_delete_stream(self, data: dict) -> AsyncIterator[dict]:
        object_key = data.get("object_key")
        if not object_key:
            yield self._error(
                code="OBJECT_KEY_MISSING",
                message="object_key is required",
                stage_failed="validation",
            )
            return

        yield self._progress("deleting")
        logger.info("Deleting object: s3://%s/%s", settings.YA_S3_BUCKET_NAME, object_key)

        try:
            async with self._get_s3_client() as s3_client:
                await s3_client.delete_object(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Key=object_key,
                )

            result = {
                "object_key": object_key,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            }
            yield self._success(result=result)
        except Exception as exc:
            error_message = self._handle_s3_error(exc, "delete")
            logger.exception("Delete failed for %s", object_key)
            yield self._error(
                code="DELETE_FAILED",
                message=error_message,
                stage_failed="deleting",
                details=str(exc),
            )

    async def _execute_list_stream(self, data: dict) -> AsyncIterator[dict]:
        prefix = data.get("prefix", "")

        yield self._progress("listing")
        logger.info("Listing objects in bucket %s with prefix '%s'", settings.YA_S3_BUCKET_NAME, prefix)

        try:
            async with self._get_s3_client() as s3_client:
                paginator = s3_client.get_paginator("list_objects_v2")
                objects = []
                async for page in paginator.paginate(
                    Bucket=settings.YA_S3_BUCKET_NAME,
                    Prefix=prefix,
                ):
                    if "Contents" not in page:
                        continue
                    for obj in page["Contents"]:
                        objects.append(
                            {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"].isoformat(),
                                "etag": obj["ETag"].strip('"'),
                            }
                        )

            result = {
                "bucket": settings.YA_S3_BUCKET_NAME,
                "prefix": prefix,
                "count": len(objects),
                "objects": objects,
            }
            yield self._success(result=result)
        except Exception as exc:
            error_message = self._handle_s3_error(exc, "list")
            logger.exception("List failed for prefix '%s'", prefix)
            yield self._error(
                code="LIST_FAILED",
                message=error_message,
                stage_failed="listing",
                details=str(exc),
            )

    async def generate_presigned_url(self, object_key: str, expiration_hours: Optional[int] = None) -> str:
        if expiration_hours is None:
            expiration_hours = settings.YA_S3_SIGNED_URL_EXPIRATION_HOURS

        expiration_seconds = expiration_hours * 3600

        async with self._get_s3_client() as s3_client:
            return await s3_client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": settings.YA_S3_BUCKET_NAME,
                    "Key": object_key,
                },
                ExpiresIn=expiration_seconds,
            )

    def execute(self, data: dict) -> dict:
        logger.warning("YaS3Service.execute() called - prefer execute_stream()")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            messages: list[dict] = []

            async def collect_messages():
                async for message in self.execute_stream(data):
                    messages.append(message)

            loop.run_until_complete(collect_messages())
            return messages[-1] if messages else self._error(
                code="SERVICE_ERROR",
                message="No response",
                stage_failed="execution",
            )
        finally:
            loop.close()
