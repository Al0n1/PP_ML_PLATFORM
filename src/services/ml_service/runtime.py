from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from src.config.services.ml_config import settings

logger = logging.getLogger(__name__)

_COMMON_BINARY_DIRS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/opt/local/bin",
)
_RUNTIME_CONFIGURED = False


def _candidate_from_value(value: str, binary_name: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    path = Path(candidate).expanduser()
    if path.is_dir():
        path = path / binary_name
    if path.is_file():
        return str(path.resolve())

    return None


def _resolve_binary(binary_name: str, *candidates: str) -> str | None:
    seen: set[str] = set()

    for candidate in candidates:
        resolved = _candidate_from_value(candidate, binary_name)
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        return resolved

    resolved = shutil.which(binary_name)
    if resolved:
        return resolved

    for directory in _COMMON_BINARY_DIRS:
        fallback = Path(directory) / binary_name
        if fallback.is_file():
            return str(fallback.resolve())

    return None


def configure_media_binaries() -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return

    ffmpeg_binary = _resolve_binary(
        "ffmpeg",
        settings.FFMPEG_BINARY,
        os.environ.get("PP_ML_FFMPEG_BIN", ""),
        os.environ.get("FFMPEG_BINARY", ""),
        os.environ.get("IMAGEIO_FFMPEG_EXE", ""),
    )

    if not ffmpeg_binary:
        logger.warning(
            "ffmpeg binary was not found. Set FFMPEG_BINARY in .env or export PP_ML_FFMPEG_BIN.",
        )
        _RUNTIME_CONFIGURED = True
        return

    ffmpeg_dir = str(Path(ffmpeg_binary).parent)
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []
    if ffmpeg_dir not in path_parts:
        os.environ["PATH"] = os.pathsep.join([ffmpeg_dir, *path_parts]) if path_parts else ffmpeg_dir

    os.environ["PP_ML_FFMPEG_BIN"] = ffmpeg_binary
    os.environ["FFMPEG_BINARY"] = ffmpeg_binary
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_binary

    ffprobe_binary = _resolve_binary(
        "ffprobe",
        settings.FFPROBE_BINARY,
        os.environ.get("PP_ML_FFPROBE_BIN", ""),
        os.environ.get("FFPROBE_BINARY", ""),
        str(Path(ffmpeg_binary).with_name("ffprobe")),
    )
    if ffprobe_binary:
        os.environ["PP_ML_FFPROBE_BIN"] = ffprobe_binary
        os.environ["FFPROBE_BINARY"] = ffprobe_binary

    try:
        from pydub import AudioSegment
    except Exception:
        logger.debug("pydub is not available during ffmpeg runtime configuration", exc_info=True)
    else:
        AudioSegment.converter = ffmpeg_binary
        AudioSegment.ffmpeg = ffmpeg_binary
        if ffprobe_binary:
            AudioSegment.ffprobe = ffprobe_binary

    logger.info("Configured media binaries: ffmpeg=%s ffprobe=%s", ffmpeg_binary, ffprobe_binary or "")
    _RUNTIME_CONFIGURED = True
