"""
HTTP transport layer for file processing.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.sse_utils import get_sse_headers

router = APIRouter(prefix='/files', tags=["files"])
logger = logging.getLogger(__name__)


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
    logger.info("SSE processing started for URL: %s", request_body.download_url)
    pipeline = request.app.state.file_pipeline
    return StreamingResponse(
        pipeline.process_stream(request_body.download_url),
        media_type="text/event-stream",
        headers=get_sse_headers(),
    )
