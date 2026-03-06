from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config.app_config import settings as app_settings
from src.config.logging_config import setup_logging
from src.config.services.ml_config import settings as ml_settings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _cli_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ml_service directly for a single input file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Absolute path or path relative to the current working directory.",
    )
    return parser


def resolve_source_path(cli_source: str | None) -> Path:
    if cli_source:
        return _cli_path(cli_source)

    env_source = ml_settings.ML_CLI_SOURCE_PATH.strip()
    if not env_source:
        raise ValueError(
            "Source path is not configured. Set ML_CLI_SOURCE_PATH in .env or pass --source.",
        )
    return _project_path(env_source)


def resolve_output_path(source_path: Path) -> Path:
    temp_dir = _project_path(ml_settings.TEMP_DIR)
    return (temp_dir / f"{source_path.stem}.mp4").resolve()


def configure_logging() -> None:
    log_dir = _project_path(app_settings.LOG_DIR)
    setup_logging(log_level=app_settings.LOG_LEVEL, log_dir=str(log_dir))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging()

    try:
        source_path = resolve_source_path(args.source)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if not source_path.is_file():
        logger.error("ML CLI source file not found: %s", source_path)
        return 1

    logger.info("Starting ml_service CLI for source: %s", source_path)

    from src.services.ml_service.ml_service import MLService

    ml_service = MLService()
    result = ml_service.execute({"path": str(source_path)})

    if result.get("status") != "success":
        logger.error(
            "ml_service processing failed: %s",
            result.get("message", "Unknown error"),
        )
        return 1

    output_path = resolve_output_path(source_path)
    if not output_path.is_file():
        logger.error("ml_service reported success, but output file not found: %s", output_path)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
