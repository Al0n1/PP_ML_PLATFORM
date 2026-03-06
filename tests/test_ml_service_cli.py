from __future__ import annotations

import io
import logging
import shutil
import sys
import types
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import src.command.ml_service as cli_module

TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / "var" / "temp" / "test_cli"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _configure_stdout_logging(*, log_level: str, log_dir: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


@contextmanager
def _workspace_tempdir():
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class MLServiceCLITestCase(unittest.TestCase):
    def test_env_source_path_is_resolved_from_project_root(self) -> None:
        with _workspace_tempdir() as project_root:
            input_path = project_root / "fixtures" / "sample.mp4"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(b"video")

            temp_output_dir = project_root / "var" / "temp"
            log_dir = project_root / "var" / "log"
            recorded_path: dict[str, str] = {}

            class FakeMLService:
                def execute(self, data):
                    recorded_path["path"] = data["path"]
                    output_path = temp_output_dir / "sample.mp4"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"processed")
                    logging.getLogger("fake.ml_service").info("service log line")
                    return {"status": "success", "result": {"output": "sample.mp4"}}

            fake_module = types.SimpleNamespace(MLService=FakeMLService)

            stdout = io.StringIO()
            with (
                patch.object(cli_module, "PROJECT_ROOT", project_root),
                patch.object(
                    cli_module,
                    "ml_settings",
                    SimpleNamespace(
                        ML_CLI_SOURCE_PATH="fixtures/sample.mp4",
                        TEMP_DIR="var/temp",
                    ),
                ),
                patch.object(
                    cli_module,
                    "app_settings",
                    SimpleNamespace(LOG_LEVEL="INFO", LOG_DIR=str(log_dir)),
                ),
                patch.object(cli_module, "setup_logging", side_effect=_configure_stdout_logging),
                patch.dict(sys.modules, {"src.services.ml_service.ml_service": fake_module}),
                redirect_stdout(stdout),
            ):
                exit_code = cli_module.main([])

            self.assertEqual(exit_code, 0)
            self.assertEqual(recorded_path["path"], str(input_path.resolve()))
            output_lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
            self.assertIn("service log line", output_lines)
            self.assertEqual(output_lines[-1], str((temp_output_dir / "sample.mp4").resolve()))

    def test_cli_source_overrides_env_value(self) -> None:
        with _workspace_tempdir() as project_root:
            cli_input = project_root / "cli-input.mp4"
            cli_input.write_bytes(b"video")

            temp_output_dir = project_root / "var" / "temp"
            observed: dict[str, str] = {}

            class FakeMLService:
                def execute(self, data):
                    observed["path"] = data["path"]
                    output_path = temp_output_dir / "cli-input.mp4"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"processed")
                    return {"status": "success", "result": {"output": "cli-input.mp4"}}

            fake_module = types.SimpleNamespace(MLService=FakeMLService)

            with (
                patch.object(cli_module, "PROJECT_ROOT", project_root),
                patch.object(
                    cli_module,
                    "ml_settings",
                    SimpleNamespace(
                        ML_CLI_SOURCE_PATH="fixtures/from-env.mp4",
                        TEMP_DIR="var/temp",
                    ),
                ),
                patch.object(
                    cli_module,
                    "app_settings",
                    SimpleNamespace(LOG_LEVEL="INFO", LOG_DIR="var/log"),
                ),
                patch.object(cli_module, "setup_logging", side_effect=_configure_stdout_logging),
                patch.dict(sys.modules, {"src.services.ml_service.ml_service": fake_module}),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = cli_module.main(["--source", str(cli_input)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(observed["path"], str(cli_input.resolve()))

    def test_missing_source_file_returns_error(self) -> None:
        with _workspace_tempdir() as project_root:
            stdout = io.StringIO()

            with (
                patch.object(cli_module, "PROJECT_ROOT", project_root),
                patch.object(
                    cli_module,
                    "ml_settings",
                    SimpleNamespace(
                        ML_CLI_SOURCE_PATH="fixtures/missing.mp4",
                        TEMP_DIR="var/temp",
                    ),
                ),
                patch.object(
                    cli_module,
                    "app_settings",
                    SimpleNamespace(LOG_LEVEL="INFO", LOG_DIR="var/log"),
                ),
                patch.object(cli_module, "setup_logging", side_effect=_configure_stdout_logging),
                redirect_stdout(stdout),
            ):
                exit_code = cli_module.main([])

            self.assertEqual(exit_code, 1)
            self.assertIn("ML CLI source file not found", stdout.getvalue())

    def test_missing_env_source_returns_error(self) -> None:
        stdout = io.StringIO()

        with (
            patch.object(cli_module, "PROJECT_ROOT", Path.cwd()),
            patch.object(
                cli_module,
                "ml_settings",
                SimpleNamespace(
                    ML_CLI_SOURCE_PATH="",
                    TEMP_DIR="var/temp",
                ),
            ),
            patch.object(
                cli_module,
                "app_settings",
                SimpleNamespace(LOG_LEVEL="INFO", LOG_DIR="var/log"),
            ),
            patch.object(cli_module, "setup_logging", side_effect=_configure_stdout_logging),
            redirect_stdout(stdout),
        ):
            exit_code = cli_module.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Source path is not configured", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
