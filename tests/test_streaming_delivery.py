from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import src.routers.file_router as file_router_module
import src.services.ml_service.ml_service as ml_module


def _response(*, result=None, error=None):
    return SimpleNamespace(status=error is None, result=result, error=error)


class _FakeOCR:
    def batch(self, images):
        return _response(result=[{"image": image} for image in images])

    def ocr_to_dict(self, raw):
        return raw


class _TestMLService(ml_module.MLService):
    def _init_models(self):
        self.translator = SimpleNamespace(translate=lambda transcript: _response(result="translated"))
        self.recognizer = SimpleNamespace(transcribe=lambda path: _response(result="transcript"))
        self.generator = SimpleNamespace(
            synthesize=lambda text, output_path=None: _response(result={"output_path": output_path})
        )
        self.ocr = _FakeOCR()


async def _upstream_sse_chunks(observed: list[str]):
    observed.append("upstream:first")
    yield "event: progress\ndata: {\"progress\": 5, \"stage\": \"downloading\"}\n\n"
    observed.append("upstream:second")
    yield "event: complete\ndata: {\"progress\": 100, \"stage\": \"complete\"}\n\n"


@pytest.mark.asyncio
async def test_stream_pipeline_events_yields_checkpoint_before_next_upstream_chunk() -> None:
    observed: list[str] = []
    wrapped_stream = file_router_module._stream_pipeline_events(_upstream_sse_chunks(observed))

    first_chunk = await anext(wrapped_stream)

    assert first_chunk.startswith("event: progress")
    assert observed == ["upstream:first"]

    asyncio.get_running_loop().call_soon(observed.append, "checkpoint")
    second_chunk_task = asyncio.create_task(anext(wrapped_stream))

    await asyncio.sleep(0)

    assert observed == ["upstream:first", "checkpoint"]

    second_chunk = await second_chunk_task

    assert second_chunk.startswith("event: complete")
    assert observed == ["upstream:first", "checkpoint", "upstream:second"]


@pytest.mark.asyncio
async def test_ml_service_streaming_keeps_event_loop_responsive_during_blocking_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source_path = tmp_path / "sample.mp4"
    source_path.write_bytes(b"video")

    def slow_copy(src, dst):
        time.sleep(0.05)
        return str(dst)

    monkeypatch.setattr(ml_module.shutil, "copy", slow_copy)
    monkeypatch.setattr(ml_module.service_utils, "extract_frames", lambda path, output_dir: _response())
    monkeypatch.setattr(ml_module.service_utils, "get_image_paths", lambda frames_dir: ["frame_000001.jpg"])
    monkeypatch.setattr(ml_module.service_utils, "extract_audio", lambda path, audio_path: _response())
    monkeypatch.setattr(ml_module.service_utils, "wav_to_mp3", lambda wav_path, mp3_path: _response())
    monkeypatch.setattr(
        ml_module.service_utils,
        "translate_ocr_results",
        lambda translator, data: _response(result=[{"translated": True} for _ in data]),
    )
    monkeypatch.setattr(ml_module.service_utils, "translate_images", lambda *args, **kwargs: _response())
    monkeypatch.setattr(ml_module.service_utils, "create_video_with_new_audio", lambda **kwargs: _response())

    service = _TestMLService(temp_dir=str(tmp_path))
    stream = service.execute_stream(
        {
            "path": str(source_path),
            "_trace": {"trace_id": "test-trace"},
        }
    )

    first_message = await anext(stream)
    assert first_message["stage"] == "copying_file"

    checkpoints: list[str] = []
    asyncio.get_running_loop().call_soon(checkpoints.append, "scheduled")
    second_message_task = asyncio.create_task(anext(stream))

    await asyncio.sleep(0.01)

    assert checkpoints == ["scheduled"]
    assert second_message_task.done() is False

    second_message = await second_message_task
    assert second_message["stage"] == "splitting_frames"

    await stream.aclose()
