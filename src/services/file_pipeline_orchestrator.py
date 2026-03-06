"""
Orchestrates the file download -> ML processing -> upload pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
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

STREAM_LAG_WARNING_MS = 5_000


@dataclass(frozen=True)
class PipelineTraceContext:
    trace_id: str
    client_host: str | None
    object_key: str | None
    request_started_monotonic: float
    request_started_at: datetime


class FilePipelineOrchestrator:
    def __init__(self, ml_service: MLService, s3_service: YaS3Service, temp_dir: str | None = None):
        self.ml_service = ml_service
        self.s3_service = s3_service
        self.formatter = SSEEventFormatter()
        self.temp_dir = Path(temp_dir or ml_settings.TEMP_DIR)
        self.download_dir = self.temp_dir / "downloads"

    async def process_stream(
        self,
        download_url: str,
        *,
        trace_id: str | None = None,
        client_host: str | None = None,
    ) -> AsyncIterator[str]:
        downloaded_file_path: Optional[Path] = None
        output_file_path: Optional[Path] = None

        object_key = self.extract_object_key_from_url(download_url)
        trace = PipelineTraceContext(
            trace_id=trace_id or uuid.uuid4().hex[:12],
            client_host=client_host,
            object_key=object_key,
            request_started_monotonic=perf_counter(),
            request_started_at=datetime.now(timezone.utc),
        )
        sanitized_url = self._sanitize_url(download_url)
        logger.info(
            "[trace=%s] pipeline.start client=%s object_key=%s presigned=%s url=%s",
            trace.trace_id,
            trace.client_host or "-",
            object_key or "-",
            self.is_presigned_url(download_url),
            sanitized_url,
        )

        try:
            if not object_key:
                async for event in self._yield_formatted(
                    trace,
                    "pipeline.validation",
                    build_error(
                        code="INVALID_URL",
                        message="Failed to extract object key from URL",
                        stage_failed="url_parsing",
                    ),
                ):
                    yield event
                return

            try:
                if self.is_presigned_url(download_url):
                    async for event in self._yield_formatted(
                        trace,
                        "pipeline.download",
                        build_progress(progress=5, stage="downloading"),
                    ):
                        yield event

                    downloaded_file_path, file_size = await self._download_file_from_url(
                        download_url,
                        object_key,
                        trace=trace,
                    )

                    async for event in self._yield_formatted(
                        trace,
                        "pipeline.download",
                        build_progress(
                            progress=15,
                            stage="downloading",
                            details={"size": file_size},
                        ),
                    ):
                        yield event
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
                        self._log_upstream_message(trace, "s3.download", message)
                        if self._is_error(message):
                            async for event in self._yield_formatted(trace, "s3.download", message):
                                yield event
                            return
                        if self._is_complete(message):
                            download_result = message.get("result")
                            continue
                        async for event in self._yield_formatted(trace, "s3.download", message):
                            yield event

                    if not download_result:
                        async for event in self._yield_formatted(
                            trace,
                            "s3.download",
                            build_error(
                                code="DOWNLOAD_FAILED",
                                message="S3 service did not return a download result",
                                stage_failed="download",
                            ),
                        ):
                            yield event
                        return

                    downloaded_file_path = Path(download_result["download_path"])

                if downloaded_file_path is None or not downloaded_file_path.exists():
                    async for event in self._yield_formatted(
                        trace,
                        "pipeline.download",
                        build_error(
                            code="DOWNLOAD_FAILED",
                            message="Failed to download file",
                            stage_failed="download",
                        ),
                    ):
                        yield event
                    return

                ml_result = None
                file_name = downloaded_file_path.name
                file_name_without_ext = downloaded_file_path.stem

                logger.info(
                    "[trace=%s] pipeline.ml.start input_path=%s file_name=%s",
                    trace.trace_id,
                    downloaded_file_path,
                    file_name,
                )
                async for message in self.ml_service.execute_stream(
                    {
                        "data": {
                            "name": file_name_without_ext,
                            "path": str(downloaded_file_path),
                            "_trace": {"trace_id": trace.trace_id},
                        }
                    }
                ):
                    self._log_upstream_message(trace, "ml_service", message)
                    if self._is_error(message):
                        async for event in self._yield_formatted(trace, "ml_service", message):
                            yield event
                        return
                    if self._is_complete(message):
                        ml_result = message.get("result")
                        continue
                    async for event in self._yield_formatted(trace, "ml_service", message):
                        yield event

                if not ml_result or "output" not in ml_result:
                    async for event in self._yield_formatted(
                        trace,
                        "ml_service",
                        build_error(
                            code="ML_PROCESSING_FAILED",
                            message="ML service did not return a processing result",
                            stage_failed="ml_processing",
                        ),
                    ):
                        yield event
                    return

                output_file_path = self.temp_dir / ml_result["output"]
                if not output_file_path.exists():
                    async for event in self._yield_formatted(
                        trace,
                        "ml_service",
                        build_error(
                            code="ML_PROCESSING_FAILED",
                            message="ML output file is missing",
                            stage_failed="ml_processing",
                        ),
                    ):
                        yield event
                    return

                upload_result = None
                logger.info(
                    "[trace=%s] pipeline.upload.start output_path=%s object_key=%s",
                    trace.trace_id,
                    output_file_path,
                    output_file_path.name,
                )
                async for message in self.s3_service.execute_stream(
                    {
                        "data": {
                            "operation": "upload",
                            "file_path": str(output_file_path),
                            "object_key": output_file_path.name,
                        }
                    }
                ):
                    self._log_upstream_message(trace, "s3.upload", message)
                    if self._is_error(message):
                        async for event in self._yield_formatted(trace, "s3.upload", message):
                            yield event
                        return
                    if self._is_complete(message):
                        upload_result = message.get("result")
                        continue
                    async for event in self._yield_formatted(trace, "s3.upload", message):
                        yield event

                if not upload_result:
                    async for event in self._yield_formatted(
                        trace,
                        "s3.upload",
                        build_error(
                            code="UPLOAD_FAILED",
                            message="S3 service did not return an upload result",
                            stage_failed="upload",
                        ),
                    ):
                        yield event
                    return

                final_result = {
                    "download_url": upload_result.get("download_url"),
                    "object_key": upload_result.get("object_key"),
                    "size": upload_result.get("size"),
                    "original_file": object_key,
                    "url_expires_in_hours": upload_result.get("url_expires_in_hours"),
                }
                async for event in self._yield_formatted(
                    trace,
                    "pipeline.complete",
                    build_success(result=final_result),
                ):
                    yield event

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "[trace=%s] HTTP download failed with status %s",
                    trace.trace_id,
                    exc.response.status_code,
                    exc_info=True,
                )
                async for event in self._yield_formatted(
                    trace,
                    "pipeline.http_download",
                    build_error(
                        code="HTTP_DOWNLOAD_FAILED",
                        message=f"HTTP download failed with status {exc.response.status_code}",
                        stage_failed="http_download",
                        details=str(exc),
                    ),
                ):
                    yield event
            except httpx.HTTPError as exc:
                logger.error("[trace=%s] HTTP download failed", trace.trace_id, exc_info=True)
                async for event in self._yield_formatted(
                    trace,
                    "pipeline.http_download",
                    build_error(
                        code="HTTP_DOWNLOAD_FAILED",
                        message="HTTP download failed",
                        stage_failed="http_download",
                        details=str(exc),
                    ),
                ):
                    yield event
            except Exception as exc:
                logger.error("[trace=%s] Critical error in file pipeline", trace.trace_id, exc_info=True)
                async for event in self._yield_formatted(
                    trace,
                    "pipeline.internal_error",
                    build_error(
                        code="INTERNAL_ERROR",
                        message="Critical server error",
                        stage_failed="streaming",
                        details=str(exc),
                        recoverable=False,
                    ),
                ):
                    yield event
        finally:
            await self._cleanup_local_files(downloaded_file_path, output_file_path)
            logger.info(
                "[trace=%s] pipeline.finish elapsed_ms=%s downloaded_path=%s output_path=%s",
                trace.trace_id,
                int((perf_counter() - trace.request_started_monotonic) * 1000),
                downloaded_file_path,
                output_file_path,
            )

    def _format(self, message: dict) -> str:
        return self.formatter.format_event(message)

    async def _yield_formatted(
        self,
        trace: PipelineTraceContext,
        source: str,
        message: dict,
    ) -> AsyncIterator[str]:
        formatted = self._format(message)
        payload_bytes = len(formatted.encode("utf-8"))
        self._log_message_event(
            trace,
            source,
            "before_stream_yield",
            message,
            payload_bytes=payload_bytes,
        )
        yield_started = perf_counter()
        yield formatted
        self._log_message_event(
            trace,
            source,
            "after_stream_resume",
            message,
            payload_bytes=payload_bytes,
            resume_gap_ms=int((perf_counter() - yield_started) * 1000),
        )

    def _log_upstream_message(self, trace: PipelineTraceContext, source: str, message: dict) -> None:
        self._log_message_event(trace, source, "upstream_received", message)

    def _log_message_event(
        self,
        trace: PipelineTraceContext,
        source: str,
        boundary: str,
        message: dict,
        *,
        payload_bytes: int | None = None,
        resume_gap_ms: int | None = None,
    ) -> None:
        lag_ms = self._message_lag_ms(message)
        elapsed_ms = int((perf_counter() - trace.request_started_monotonic) * 1000)
        level = logging.DEBUG
        if (lag_ms is not None and lag_ms >= STREAM_LAG_WARNING_MS) or (
            resume_gap_ms is not None and resume_gap_ms >= STREAM_LAG_WARNING_MS
        ):
            level = logging.WARNING
        logger.log(
            level,
            "[trace=%s] %s source=%s stage=%s progress=%s status=%s msg_ts=%s lag_ms=%s elapsed_ms=%s resume_gap_ms=%s payload_bytes=%s details=%s",
            trace.trace_id,
            boundary,
            source,
            message.get("stage"),
            message.get("progress"),
            message.get("status"),
            message.get("timestamp"),
            lag_ms,
            elapsed_ms,
            resume_gap_ms,
            payload_bytes,
            self._summarize_details(message.get("details")),
        )

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

    @staticmethod
    def _sanitize_url(url: str) -> str:
        parsed = urlparse(url)
        sanitized = parsed._replace(query="", fragment="")
        return sanitized.geturl()

    @staticmethod
    def _message_lag_ms(message: dict) -> int | None:
        timestamp = message.get("timestamp")
        if not timestamp:
            return None
        try:
            event_time = datetime.fromisoformat(str(timestamp))
        except ValueError:
            return None
        return int((datetime.now(timezone.utc) - event_time).total_seconds() * 1000)

    @staticmethod
    def _summarize_details(details: object) -> str | None:
        if details is None:
            return None
        if isinstance(details, dict):
            current_step = details.get("current_step")
            total_steps = details.get("total_steps")
            if current_step is not None and total_steps is not None:
                return f"{current_step}/{total_steps}"
        return str(details)

    async def _download_file_from_url(
        self,
        url: str,
        object_key: str,
        *,
        trace: PipelineTraceContext | None = None,
    ) -> tuple[Path, int]:
        destination = self.download_dir / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        download_started = perf_counter()
        if trace is not None:
            logger.info(
                "[trace=%s] download.start object_key=%s destination=%s url=%s",
                trace.trace_id,
                object_key,
                destination,
                self._sanitize_url(url),
            )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                downloaded_size = 0
                chunk_count = 0
                first_chunk_logged = False
                if trace is not None:
                    logger.info(
                        "[trace=%s] download.headers status=%s content_length=%s content_type=%s",
                        trace.trace_id,
                        response.status_code,
                        response.headers.get("content-length"),
                        response.headers.get("content-type"),
                    )
                async with aiofiles.open(destination, "wb") as file_obj:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        chunk_count += 1
                        await file_obj.write(chunk)
                        downloaded_size += len(chunk)
                        if trace is not None and not first_chunk_logged:
                            first_chunk_logged = True
                            logger.info(
                                "[trace=%s] download.first_chunk size=%s elapsed_ms=%s",
                                trace.trace_id,
                                len(chunk),
                                int((perf_counter() - download_started) * 1000),
                            )

        if trace is not None:
            logger.info(
                "[trace=%s] download.complete bytes=%s chunks=%s duration_ms=%s destination=%s",
                trace.trace_id,
                downloaded_size,
                chunk_count,
                int((perf_counter() - download_started) * 1000),
                destination,
            )

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
