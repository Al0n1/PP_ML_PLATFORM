#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
TARGET_BRANCH="${2:-main}"

REPO_DIR="${PP_ML_REPO_DIR:-$HOME/PP_ML_PLATFORM}"
VENV_DIR="${PP_ML_VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PP_ML_PYTHON_BIN:-$VENV_DIR/bin/python}"
CLI_MODULE="${PP_ML_CLI_MODULE:-src.command.ml_service}"
ENV_FILE="${PP_ML_ENV_FILE:-$REPO_DIR/.env}"

emit_error() {
    printf 'MLRUN_ERROR %s\n' "$1" >&2
}

require_repo() {
    if [[ ! -d "$REPO_DIR" ]]; then
        emit_error "Repository directory does not exist: $REPO_DIR"
        exit 1
    fi
    cd "$REPO_DIR"
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        emit_error "Directory is not a git repository: $REPO_DIR"
        exit 1
    fi
}

activate_python() {
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_DIR/bin/activate"
    fi
    if [[ ! -x "$PYTHON_BIN" ]]; then
        emit_error "Python binary is not executable: $PYTHON_BIN"
        exit 1
    fi
    export PYTHONUNBUFFERED=1
}

ensure_clean_worktree() {
    if ! git diff --quiet || ! git diff --cached --quiet; then
        emit_error "Working tree is dirty. Commit or stash local changes before running."
        exit 1
    fi
}

resolve_input_path() {
    "$PYTHON_BIN" -c "from src.command.ml_service import resolve_source_path; print(resolve_source_path(None))"
}

resolve_output_path() {
    "$PYTHON_BIN" -c "from src.command.ml_service import resolve_source_path, resolve_output_path; print(resolve_output_path(resolve_source_path(None)))"
}

ml_env_action() {
    local mode="$1"
    local payload_json=""

    require_repo
    activate_python

    if [[ "$mode" == "update" ]]; then
        payload_json="$(cat)"
    fi

    ML_ENV_MODE="$mode" \
    ML_ENV_FILE="$ENV_FILE" \
    ML_ENV_PAYLOAD="$payload_json" \
    "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

ENV_PREFIX = "MLRUN_ENV "
START_PREFIX = "###<"
END_PREFIX = "###>"
DEFAULT_START = "###< ML Service"
DEFAULT_END = "###>"

mode = os.environ["ML_ENV_MODE"]
env_path = Path(os.environ["ML_ENV_FILE"])
payload = json.loads(os.environ.get("ML_ENV_PAYLOAD") or "{}")
updates = payload.get("params") or {}

if not env_path.exists():
    raise SystemExit(f"Env file does not exist: {env_path}")

text = env_path.read_text(encoding="utf-8")
lines = text.splitlines(keepends=True)

start_idx = None
end_idx = None
for index, line in enumerate(lines):
    stripped = line.strip()
    if start_idx is None and stripped.startswith(START_PREFIX) and "ML" in stripped.upper():
        start_idx = index
        continue
    if start_idx is not None and index > start_idx and stripped.startswith(END_PREFIX):
        end_idx = index
        break

if start_idx is None or end_idx is None:
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if lines and lines[-1].strip():
        lines.append("\n")
    start_idx = len(lines)
    lines.extend([DEFAULT_START + "\n", DEFAULT_END + "\n"])
    end_idx = len(lines) - 1

body = lines[start_idx + 1:end_idx]
key_positions: dict[str, int] = {}
params: dict[str, str] = {}

for index, line in enumerate(body):
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        continue
    key_positions[key] = index
    params[key] = value.rstrip("\r\n")

if mode == "update":
    for key, value in updates.items():
        rendered = f"{key}={value}\n"
        if key in key_positions:
            body[key_positions[key]] = rendered
        else:
            body.append(rendered)
        params[key] = value

    lines[start_idx + 1:end_idx] = body
    env_path.write_text("".join(lines), encoding="utf-8")

block_text = "".join(lines[start_idx:end_idx + 1]).rstrip("\n")
print(
    ENV_PREFIX
    + json.dumps(
        {
            "env_file": str(env_path),
            "block": block_text,
            "params": params,
        },
        ensure_ascii=False,
    )
)
PY
}

run_health() {
    require_repo
    activate_python

    local current_sha
    local git_ok=false
    local python_ok=false
    local cli_ok=false
    local input_file_ok=false
    local input_path=""

    current_sha="$(git rev-parse HEAD 2>/dev/null || true)"

    if git ls-remote --exit-code origin "$TARGET_BRANCH" >/dev/null 2>&1; then
        git_ok=true
    fi

    if "$PYTHON_BIN" -c "import src.command.ml_service" >/dev/null 2>&1; then
        python_ok=true
    fi

    if "$PYTHON_BIN" -m "$CLI_MODULE" --help >/dev/null 2>&1; then
        cli_ok=true
    fi

    input_path="$(resolve_input_path 2>/dev/null || true)"
    if [[ -n "$input_path" && -f "$input_path" ]]; then
        input_file_ok=true
    fi

    printf 'MLRUN_HEALTH {"repo_ok":true,"git_ok":%s,"python_ok":%s,"cli_ok":%s,"input_file_ok":%s,"current_sha":"%s"}\n' \
        "$git_ok" \
        "$python_ok" \
        "$cli_ok" \
        "$input_file_ok" \
        "$current_sha"

    if [[ "$git_ok" == "true" && "$python_ok" == "true" && "$cli_ok" == "true" && "$input_file_ok" == "true" ]]; then
        return 0
    fi
    return 1
}

run_pipeline() {
    require_repo
    activate_python
    ensure_clean_worktree

    local git_sha_before
    local git_sha_after

    git_sha_before="$(git rev-parse HEAD)"
    git fetch origin "$TARGET_BRANCH"
    git pull --ff-only origin "$TARGET_BRANCH"
    git_sha_after="$(git rev-parse HEAD)"

    printf 'MLRUN_META {"git_sha_before":"%s","git_sha_after":"%s","target_branch":"%s"}\n' \
        "$git_sha_before" \
        "$git_sha_after" \
        "$TARGET_BRANCH"

    "$PYTHON_BIN" -m "$CLI_MODULE"
    local cli_status=$?

    if [[ $cli_status -eq 0 ]]; then
        printf 'MLRUN_OUTPUT %s\n' "$(resolve_output_path)"
    fi

    return "$cli_status"
}

case "$ACTION" in
    health)
        run_health
        ;;
    run)
        run_pipeline
        ;;
    env-get)
        ml_env_action get
        ;;
    env-update)
        ml_env_action update
        ;;
    *)
        emit_error "Unknown action: $ACTION"
        exit 1
        ;;
esac
