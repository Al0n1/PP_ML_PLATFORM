import subprocess
import numpy as np
import whisper
from typing import Union, Any


from src.config.services.ml_config import settings, Settings
from .base import BaseRecognizer
from ...utils import VideoData


class WhisperSpeechRecognitionModel(BaseRecognizer):
    def __init__(self, cache_dir=settings.MODEL_CACHE_DIR, model_name="tiny"):
        super().__init__(name=f"Whisper-{model_name}")
        self.model = whisper.load_model(model_name, download_root=cache_dir)
        
        self.license = 'MIT'

    def process(self, video_data: VideoData) -> VideoData:
        result = self.model.transcribe(video_data.source_path)
        video_data.audio.source_text = result.get("text", None)
        return video_data


# SAMPLE_RATE = 16000


# def load_audio_ffmpeg(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
#     """
#     Извлекает аудио из файла через ffmpeg в формате float32 numpy-массива.

#     Команда конвертирует любой медиафайл (видео/аудио) в:
#       - моно (-ac 1)
#       - PCM signed 16-bit little-endian (-acodec pcm_s16le)
#       - заданный sample rate (-ar sr)
#     и читает результат из stdout.

#     Returns:
#         np.ndarray dtype=float32, shape=(n_samples,), значения в [-1, 1].
#     """
#     cmd = [
#         "ffmpeg",
#         "-nostdin",
#         "-threads", "0",
#         "-i", file,
#         "-f", "s16le",
#         "-ac", "1",
#         "-acodec", "pcm_s16le",
#         "-ar", str(sr),
#         "-",
#     ]
#     process = subprocess.run(cmd, capture_output=True)
#     if process.returncode != 0:
#         raise RuntimeError(f"ffmpeg error: {process.stderr.decode()}")

#     audio = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32) / 32768.0
#     return audio


# video_path = 'var/data_ocr/small_sample.mp4'

# audio = load_audio_ffmpeg(video_path)
# print(f"shape: {audio.shape}, dtype: {audio.dtype}")
# print(f"duration: {len(audio) / SAMPLE_RATE:.2f}s")
# print(audio)

# whisper_mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio))
# print(f"mel spectrogram shape: {whisper_mel.shape}")