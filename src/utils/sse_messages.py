"""
Minimal builders for internal SSE payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_progress(progress: int, stage: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "progress": progress,
        "stage": stage,
        "status": "processing",
        "timestamp": _timestamp(),
    }
    if details:
        message["details"] = details
    return message


def build_success(result: dict[str, Any] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "progress": 100,
        "stage": "complete",
        "status": "success",
        "timestamp": _timestamp(),
    }
    if result:
        message["result"] = result
    return message


def build_error(
    code: str,
    message: str,
    stage_failed: str,
    details: str | None = None,
    recoverable: bool = True,
) -> dict[str, Any]:
    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "stage_failed": stage_failed,
        "recoverable": recoverable,
    }
    if details:
        error["details"] = details

    return {
        "progress": -1,
        "stage": "error",
        "status": "error",
        "timestamp": _timestamp(),
        "error": error,
    }
