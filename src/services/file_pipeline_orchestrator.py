"""
Orchestrates the file download -> ML processing -> upload pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import AsyncIterator, Optional
from urllib.parse import parse_qs, urlparse

import aiofiles
import httpx

from src.config.services.ml_config import settings as ml_settings
from src.services.ml_service.ml_service import MLService
from src.services.ya_s3_service.ya_s3_service import YaS3Service
from src.utils.sse_formatter import SSEEventFormatter
from src.utils.sse_messages import build_error, build_progress, build_success

logger = logging.getLogger(__name__)


class FilePipelineOrchestrator:
    def __init__(self, ml_service: MLService, s3_service: YaS3Service, temp_dir: str | None = None):
        self.ml_service = ml_service
        self.s3_service = s3_service
        self.formatter = SSEEventFormatter()
        self.temp_dir = Path(temp_dir or ml_settings.TEMP_DIR)
        self.download_dir = self.temp_dir / "downloads"

    async def process_stream(self, download_url: str) -> AsyncIterator[str]:
        downloaded_file_path: Optional[Path] = None
        output_file_path: Optional[Path] = None

        object_key = self.extract_object_key_from_url(download_url)
        if not object_key:
            yield self._format(
                build_error(
                    code="INVALID_URL",
                    message="Failed to extract object key from URL",
                    stage_failed="url_parsing",
                )
            )
            return

        try:
            if self.is_presigned_url(download_url):
                yield self._format(build_progress(progress=5, stage="downloading"))
                downloaded_file_path, file_size = await self._download_file_from_url(download_url, object_key)
                yield self._format(
                    build_progress(
                        progress=15,
                        stage="downloading",
                        details={"size": file_size},
                    )
                )
            else:
                download_result = None
                download_target = self.download_dir / object_key
                async for message in self.s3_service.execute_stream(
                    {
                        "data": {
                            "operation": "download",
                            "object_key": object_key,
                            "download_path": str(download_target),
                        }
                    }
                ):
                    if self._is_error(message):
                        yield self._format(message)
                        return
                    if self._is_complete(message):
                        download_result = message.get("result")
                        continue
                    yield self._format(message)

                if not download_result:
                    yield self._format(
                        build_error(
                            code="DOWNLOAD_FAILED",
                            message="S3 service did not return a download result",
                            stage_failed="download",
                        )
                    )
                    return

                downloaded_file_path = Path(download_result["download_path"])

            if downloaded_file_path is None or not downloaded_file_path.exists():
                yield self._format(
                    build_error(
                        code="DOWNLOAD_FAILED",
                        message="Failed to download file",
                        stage_failed="download",
                    )
                )
                return

            ml_result = None
            file_name = downloaded_file_path.name
            file_name_without_ext = downloaded_file_path.stem

            async for message in self.ml_service.execute_stream(
                {
                    "data": {
                        "name": file_name_without_ext,
                        "path": str(downloaded_file_path),
                    }
                }
            ):
                if self._is_error(message):
                    yield self._format(message)
                    return
                if self._is_complete(message):
                    ml_result = message.get("result")
                    continue
                yield self._format(message)

            if not ml_result or "output" not in ml_result:
                yield self._format(
                    build_error(
                        code="ML_PROCESSING_FAILED",
                        message="ML service did not return a processing result",
                        stage_failed="ml_processing",
                    )
                )
                return

            output_file_path = self.temp_dir / ml_result["output"]
            if not output_file_path.exists():
                yield self._format(
                    build_error(
                        code="ML_PROCESSING_FAILED",
                        message="ML output file is missing",
                        stage_failed="ml_processing",
                    )
                )
                return

            upload_result = None
            async for message in self.s3_service.execute_stream(
                {
                    "data": {
                        "operation": "upload",
                        "file_path": str(output_file_path),
                        "object_key": output_file_path.name,
                    }
                }
            ):
                if self._is_error(message):
                    yield self._format(message)
                    return
                if self._is_complete(message):
                    upload_result = message.get("result")
                    continue
                yield self._format(message)

            if not upload_result:
                yield self._format(
                    build_error(
                        code="UPLOAD_FAILED",
                        message="S3 service did not return an upload result",
                        stage_failed="upload",
                    )
                )
                return

            final_result = {
                "download_url": upload_result.get("download_url"),
                "object_key": upload_result.get("object_key"),
                "size": upload_result.get("size"),
                "original_file": object_key,
                "url_expires_in_hours": upload_result.get("url_expires_in_hours"),
            }
            yield self._format(build_success(result=final_result))

        except httpx.HTTPStatusError as exc:
            logger.error("HTTP download failed with status %s", exc.response.status_code, exc_info=True)
            yield self._format(
                build_error(
                    code="HTTP_DOWNLOAD_FAILED",
                    message=f"HTTP download failed with status {exc.response.status_code}",
                    stage_failed="http_download",
                    details=str(exc),
                )
            )
        except httpx.HTTPError as exc:
            logger.error("HTTP download failed", exc_info=True)
            yield self._format(
                build_error(
                    code="HTTP_DOWNLOAD_FAILED",
                    message="HTTP download failed",
                    stage_failed="http_download",
                    details=str(exc),
                )
            )
        except Exception as exc:
            logger.error("Critical error in file pipeline", exc_info=True)
            yield self._format(
                build_error(
                    code="INTERNAL_ERROR",
                    message="Critical server error",
                    stage_failed="streaming",
                    details=str(exc),
                    recoverable=False,
                )
            )
        finally:
            await self._cleanup_local_files(downloaded_file_path, output_file_path)

    def _format(self, message: dict) -> str:
        return self.formatter.format_event(message)

    @staticmethod
    def _is_complete(message: dict) -> bool:
        return message.get("status") == "success" and message.get("stage") == "complete"

    @staticmethod
    def _is_error(message: dict) -> bool:
        return message.get("status") == "error"

    @staticmethod
    def extract_object_key_from_url(url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            path = parsed.path.lstrip("/")
            parts = path.split("/", 1)
            if len(parts) > 1:
                if ".storage." in parsed.netloc:
                    return path
                return parts[1]
            return path or None
        except Exception:
            logger.exception("Failed to extract object key from URL: %s", url)
            return None

    @staticmethod
    def is_presigned_url(url: str) -> bool:
        try:
            query_params = parse_qs(urlparse(url).query)
            signature_params = ("X-Amz-Signature", "X-Amz-Credential", "X-Amz-Algorithm")
            return any(param in query_params for param in signature_params)
        except Exception:
            return False

    async def _download_file_from_url(self, url: str, object_key: str) -> tuple[Path, int]:
        destination = self.download_dir / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                downloaded_size = 0
                async with aiofiles.open(destination, "wb") as file_obj:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        await file_obj.write(chunk)
                        downloaded_size += len(chunk)

        return destination, downloaded_size

    async def _cleanup_local_files(self, *paths: Optional[Path]) -> None:
        for path in paths:
            if path is None or not path.exists():
                continue

            try:
                if path.is_file():
                    await asyncio.to_thread(path.unlink)
                    self._cleanup_empty_parent_dirs(path.parent)
                elif path.is_dir():
                    await asyncio.to_thread(shutil.rmtree, path)
            except Exception:
                logger.warning("Failed to cleanup path: %s", path, exc_info=True)

    def _cleanup_empty_parent_dirs(self, start_dir: Path) -> None:
        current_dir = start_dir
        while current_dir != self.download_dir.parent and current_dir.exists():
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent
