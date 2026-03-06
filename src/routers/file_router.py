"""
HTTP transport layer for file processing.
"""

import asyncio
import logging
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.sse_utils import get_sse_headers

router = APIRouter(prefix='/files', tags=["files"])
logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    parsed = urlparse(url)
    sanitized = parsed._replace(query="", fragment="")
    return sanitized.geturl()


async def _stream_pipeline_events(message_stream):
    """Yield SSE chunks with an explicit checkpoint between messages."""
    async for event in message_stream:
        yield event
        await asyncio.sleep(0)


class FileProcessRequest(BaseModel):
    """Схема запроса на обработку файла."""
    download_url: str = Field(
        ..., 
        description="URL файла в S3 хранилище для скачивания и обработки"
    )


@router.post("/stream")
async def process_file_stream(
    request: Request,
    request_body: FileProcessRequest,
) -> StreamingResponse:
    trace_id = uuid.uuid4().hex[:12]
    client_host = request.client.host if request.client else "-"
    user_agent = request.headers.get("user-agent", "-")
    logger.info(
        "[trace=%s] SSE processing started client=%s user_agent=%s url=%s",
        trace_id,
        client_host,
        user_agent,
        _sanitize_url(request_body.download_url),
    )
    pipeline = request.app.state.file_pipeline
    headers = get_sse_headers()
    headers["X-Trace-Id"] = trace_id
    return StreamingResponse(
        _stream_pipeline_events(
            pipeline.process_stream(
                request_body.download_url,
                trace_id=trace_id,
                client_host=client_host,
            )
        ),
        media_type="text/event-stream",
        headers=headers,
    )
