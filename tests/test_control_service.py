from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from control_service.app import build_app
from control_service.config import ControlServiceSettings
from control_service.models import HealthCheckResult, LogEntry, MLEnvBlockResponse, RunResult

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


class FakeRunner:
    def __init__(
        self,
        *,
        health_result: HealthCheckResult | None = None,
        run_result: RunResult | None = None,
        delay: float = 0.0,
    ) -> None:
        self.health_result = health_result or HealthCheckResult(
            ssh_ok=True,
            repo_ok=True,
            git_ok=True,
            python_ok=True,
            cli_ok=True,
            input_file_ok=True,
            current_sha="abc123",
        )
        self.run_result = run_result or RunResult(
            exit_code=0,
            git_sha_before="aaa111",
            git_sha_after="bbb222",
            output_path="/tmp/output.mp4",
        )
        self.delay = delay
        self.run_calls = 0
        self.ml_env = MLEnvBlockResponse(
            env_file="/tmp/.env",
            block="###< ML Service\nTRANSLATOR_DEVICE=cpu\n###>",
            params={"TRANSLATOR_DEVICE": "cpu"},
        )

    async def run_health(self) -> HealthCheckResult:
        return self.health_result

    async def run_job(self, on_log) -> RunResult:
        self.run_calls += 1
        await on_log(LogEntry(ts="2026-03-07T00:00:00+00:00", stream="stdout", line="git pull ok"))
        if self.delay:
            await asyncio.sleep(self.delay)
        await on_log(LogEntry(ts="2026-03-07T00:00:01+00:00", stream="stdout", line="model init ok"))
        if self.run_result.exit_code != 0:
            await on_log(LogEntry(ts="2026-03-07T00:00:02+00:00", stream="stderr", line="pipeline failed"))
        return self.run_result

    async def get_ml_env(self) -> MLEnvBlockResponse:
        return self.ml_env

    async def update_ml_env(self, params: dict[str, str]) -> MLEnvBlockResponse:
        merged = dict(self.ml_env.params)
        merged.update(params)
        block_lines = ["###< ML Service", *[f"{key}={value}" for key, value in merged.items()], "###>"]
        self.ml_env = MLEnvBlockResponse(
            env_file=self.ml_env.env_file,
            block="\n".join(block_lines),
            params=merged,
        )
        return self.ml_env


class ControlServiceTestCase(unittest.TestCase):
    def _build_test_client(self, temp_dir: str, runner: FakeRunner) -> TestClient:
        settings = ControlServiceSettings(
            CONTROL_AUTH_TOKEN="test-token",
            CONTROL_SPOOL_DIR=str(Path(temp_dir) / "jobs"),
            SSH_HOST="ignored-for-tests",
        )
        app = build_app(settings=settings, runner=runner)
        return TestClient(app)

    def _wait_for_terminal_status(self, client: TestClient, job_id: str) -> dict:
        deadline = time.time() + 5
        while time.time() < deadline:
            response = client.get(f"/api/run/{job_id}", headers=AUTH_HEADERS)
            response.raise_for_status()
            payload = response.json()
            if payload["status"] in {"success", "failed"}:
                return payload
            time.sleep(0.05)
        raise AssertionError(f"Job {job_id} did not finish in time")

    def test_auth_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, FakeRunner()) as client:
                response = client.get("/api/health")

        self.assertEqual(response.status_code, 401)

    def test_health_endpoint_returns_runner_status(self) -> None:
        runner = FakeRunner(
            health_result=HealthCheckResult(
                ssh_ok=True,
                repo_ok=True,
                git_ok=True,
                python_ok=True,
                cli_ok=True,
                input_file_ok=True,
                current_sha="feedface",
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                response = client.get("/api/health", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ssh_ok"])
        self.assertEqual(payload["current_sha"], "feedface")
        self.assertIsNone(payload["last_job_status"])

    def test_update_run_status_stream_and_recent_logs(self) -> None:
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                response = client.post("/api/run/update", headers=AUTH_HEADERS)
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]

                final_payload = self._wait_for_terminal_status(client, job_id)
                self.assertEqual(final_payload["status"], "success")
                self.assertEqual(final_payload["git_sha_before"], "aaa111")
                self.assertEqual(final_payload["git_sha_after"], "bbb222")
                self.assertEqual(final_payload["output_path"], "/tmp/output.mp4")

                stream_response = client.get(f"/api/run/{job_id}/stream", headers=AUTH_HEADERS)
                self.assertEqual(stream_response.status_code, 200)
                lines = [json.loads(line) for line in stream_response.text.splitlines() if line.strip()]
                self.assertEqual(
                    [entry["line"] for entry in lines],
                    [
                        "Starting remote update and run",
                        "git pull ok",
                        "model init ok",
                        "Job finished successfully (exit_code=0)",
                    ],
                )

                recent_logs = client.get("/api/logs/recent?limit=10", headers=AUTH_HEADERS)
                self.assertEqual(recent_logs.status_code, 200)
                recent_payload = recent_logs.json()
                self.assertEqual(recent_payload[-1]["line"], "Job finished successfully (exit_code=0)")

    def test_second_update_returns_409_while_job_is_running(self) -> None:
        runner = FakeRunner(delay=0.4)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                first = client.post("/api/run/update", headers=AUTH_HEADERS)
                self.assertEqual(first.status_code, 202)
                first_job_id = first.json()["job_id"]

                second = client.post("/api/run/update", headers=AUTH_HEADERS)
                self.assertEqual(second.status_code, 409)
                self.assertEqual(second.json()["detail"]["job_id"], first_job_id)

                final_payload = self._wait_for_terminal_status(client, first_job_id)
                self.assertEqual(final_payload["status"], "success")

    def test_failed_run_sets_failed_status_and_error_summary(self) -> None:
        runner = FakeRunner(
            run_result=RunResult(
                exit_code=1,
                git_sha_before="aaa111",
                git_sha_after="aaa111",
                error_summary="Working tree is dirty",
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                response = client.post("/api/run/update", headers=AUTH_HEADERS)
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]

                final_payload = self._wait_for_terminal_status(client, job_id)
                self.assertEqual(final_payload["status"], "failed")
                self.assertEqual(final_payload["error_summary"], "Working tree is dirty")

                recent_logs = client.get("/api/logs/recent?limit=10", headers=AUTH_HEADERS)
                self.assertEqual(recent_logs.status_code, 200)
                lines = [entry["line"] for entry in recent_logs.json()]
                self.assertIn("pipeline failed", lines)
                self.assertEqual(lines[-1], "Job failed (exit_code=1): Working tree is dirty")

    def test_ml_env_get_returns_current_block(self) -> None:
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                response = client.get("/api/ml-env", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["env_file"], "/tmp/.env")
        self.assertEqual(payload["params"]["TRANSLATOR_DEVICE"], "cpu")

    def test_ml_env_put_merges_passed_params(self) -> None:
        runner = FakeRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self._build_test_client(temp_dir, runner) as client:
                response = client.put(
                    "/api/ml-env",
                    headers=AUTH_HEADERS,
                    json={
                        "params": {
                            "TRANSLATOR_DEVICE": "mps",
                            "OCR_DEVICE": "cpu",
                        }
                    },
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["params"]["TRANSLATOR_DEVICE"], "mps")
        self.assertEqual(payload["params"]["OCR_DEVICE"], "cpu")
        self.assertIn("OCR_DEVICE=cpu", payload["block"])


if __name__ == "__main__":
    unittest.main()
