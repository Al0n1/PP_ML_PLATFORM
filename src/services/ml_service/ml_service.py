"""
Сервис перевода видео.
"""
import asyncio
import os
import shutil
import logging
from datetime import datetime, timezone
from typing import AsyncIterator
from contextlib import contextmanager
from time import perf_counter
from huggingface_hub import login

from . import models
from . import utils as service_utils
from src.config.services.ml_config import settings
from src.utils.sse_messages import build_error, build_progress, build_success


STREAM_LAG_WARNING_MS = 5_000

logger = logging.getLogger(__name__)

@contextmanager
def log_duration(message: str | None = None, func_name: str | None = None):

    name = message or func_name

    # logger.info(f"⌛ {name}")
    start = perf_counter()

    try:
        yield
    finally:
        end = perf_counter()
        logger.info(f"⏱ {name} finished in {end - start:.4f} sec")

STAGE_PROGRESS = {
    "splitting_frames": 2,
    "recognizing_speech": 5,
    "translating_text": 8,
    "generating_tts": 12,
    "processing_frames": 15,
    "assembling_video": 95,
}


class MLService:
    """
    Сервис для обработки видео с применением ML-пайплайна.

    Этот класс выполняет последовательную обработку видео, включающую:
      - разбиение видео на кадры;
      - извлечение и распознавание аудио;
      - перевод текста;
      - синтез нового аудио;
      - обработку кадров;
      - сборку финального видео с обновлённым звуком и кадрами.

    Атрибуты:
        translate (Callable): Функция или объект для перевода текста.
        spech_recognize (Callable): Функция или объект для распознавания речи.
        ocr (Callable): Функция или объект для оптического распознавания текста (OCR).
        tts (Callable): Функция или объект для синтеза речи (Text-to-Speech).
        temp_dir (str): Временная директория для промежуточных файлов.
    """

    def __init__(self, temp_dir=settings.TEMP_DIR, settings=settings):
        """
        Инициализация ML-сервиса.

        Args:
            temp_dir (str): Путь к временной директории для хранения промежуточных данных.
        """
        self.audio_extract_name = 'audio_extract'
        self.audio_translate_name = 'audio_translate'
        self.audio_results_name = 'audio_results'

        self.video_ocr_name = 'video_ocr'
        self.video_translate_name = 'video_translate'
        self.stage_progress = dict(STAGE_PROGRESS)
        self.stage_order = list(self.stage_progress.keys())

        self.settings = settings

        with log_duration("INIT models"):
            self._init_models()
            
        self.temp_dir = temp_dir

        self.frame_extractor = service_utils.KeyFrameExtractor(extract_type="histogram", threshold=0.1)

    def _init_models(self):
        # Если требуется аутентификация для загрузки моделей, можно использовать токен из переменных окружения
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)

        with log_duration("Translator.__init__"):
            self.translator = models.Translator(self.settings)
        
        with log_duration("Recognizer.__init__"):
            self.recognizer = models.Recognizer(settings=self.settings)

        with log_duration("TextToSpeech.__init__"):
            self.generator = models.SpeechModel(config=self.settings)

        with log_duration("OCR.__init__"):
            self.ocr = models.OCR(config=self.settings)

    def _progress_message(
        self,
        stage_id: str,
        current_step: int | None = None,
        total_steps: int | None = None,
        trace_id: str | None = None,
    ) -> dict:
        progress = self.stage_progress[stage_id]
        details = None

        if current_step is not None and total_steps:
            current_index = self.stage_order.index(stage_id)
            next_progress = 100
            if current_index + 1 < len(self.stage_order):
                next_progress = self.stage_progress[self.stage_order[current_index + 1]]
            progress_range = max(0, next_progress - progress)
            progress = min(99, progress + int((current_step / total_steps) * progress_range))
            details = {
                "current_step": current_step,
                "total_steps": total_steps,
            }

        message = build_progress(progress=progress, stage=stage_id, details=details)
        if trace_id:
            self._log_message_created(trace_id, message)
        return message

    async def _run_blocking(self, func, /, *args, **kwargs):
        """Run blocking ML work off the event loop so SSE can flush promptly."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def execute(self, data: dict) -> dict:
        """
        Основной метод для запуска пайплайна обработки видео.

        Args:
            data (dict): Входные данные с ключами:
                - "path" (str): Путь к исходному видео.
                - "name" (str): Имя видео (без расширения).
                - "res_dir" (str, optional): Папка для сохранения результата.
                - "message" (str, optional): Служебное сообщение.

        Returns:
            dict: Результат выполнения с ключами:
                - "status" (str): Статус выполнения ("success" или "error").
                - "result" (dict): Результат обработки.
        """
        logger.info(f"MLService.execute called with data: {data}")
        payload = data.get("data", data)
        path = payload.get("path", "")
        target_language = payload.get("target_language", "ru")

        if not path:
            logger.info("Video path is missing")
            return {"status": "error", "message": "Path missing"}
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dir_path = os.path.join(self.temp_dir, name)
        os.makedirs(dir_path, exist_ok=True)
        video_data = service_utils.VideoData(source_path=path,
                                             target_language=target_language,
                                             video_name=name,
                                             output_dir=self.temp_dir,
                                             temp_dir=dir_path)

        # Запуск пайплайна обработки видео
        with log_duration(f'Обработка видео {name}'):
            resp: service_utils.Response = self.__process_video(video_data)

        if resp.status is False:
            return {"status": "error", "message": str(resp.error)}

        result = {"status": "success", "result": {"output": f"{name}.mp4"}}

        logger.info(f"MLService.execute returning: {result}")
        return result

    async def execute_stream(self, data: dict) -> AsyncIterator[dict]:
        """
        Streaming версия execute() для SSE.
        Использует тот же пайплайн на основе VideoData, что и execute(),
        но отдаёт прогресс по SSE между этапами.
        """
        dir_path = None
        payload = data.get("data", data)
        trace_id = self._extract_trace_id(payload)

        try:
            path = payload.get("path", "")
            target_language = payload.get("target_language", "ru")

            if not path:
                yield build_error(
                    code="ML_PROCESSING_FAILED",
                    message="Path missing",
                    stage_failed="validation",
                )
                return

            filename = os.path.basename(path)
            name = os.path.splitext(filename)[0]
            dir_path = os.path.join(self.temp_dir, name)
            os.makedirs(dir_path, exist_ok=True)

            video_data = service_utils.VideoData(
                source_path=path,
                target_language=target_language,
                video_name=name,
                output_dir=self.temp_dir,
            )

            logger.info("[trace=%s] ml.execute_stream.start path=%s", trace_id, path)

            # === splitting_frames ===
            yield self._progress_message("splitting_frames", trace_id=trace_id)
            video_data = await self._run_blocking(service_utils.extract_frames, video_data)
            video_data = await self._run_blocking(self.frame_extractor.process, video_data)

            # === Audio pipeline ===
            # recognizing_speech
            yield self._progress_message("recognizing_speech", trace_id=trace_id)
            video_data = await self._run_blocking(self.recognizer.process, video_data)

            # translating_text (audio)
            yield self._progress_message("translating_text", trace_id=trace_id)
            video_data = await self._run_blocking(self.translator.process, video_data)

            # generating_tts
            yield self._progress_message("generating_tts", trace_id=trace_id)
            video_data = await self._run_blocking(self.generator.process, video_data)
            resp = await self._run_blocking(
                service_utils.wav_to_mp3,
                video_data.audio.output_audio_path,
                video_data.audio.output_audio_path.replace('.wav', '.mp3'),
            )
            if not resp.status:
                raise Exception(resp.error)

            # === Video pipeline ===
            # processing_frames: OCR → translate → render
            yield self._progress_message("processing_frames", trace_id=trace_id)
            video_data = await self._run_blocking(self.ocr.process, video_data)
            video_data = await self._run_blocking(self.translator.process, video_data)

            annotations = [
                video_data.video.ocr_frames[idx]
                for idx in video_data.video.translated_frames_indexes
            ]
            resp = await self._run_blocking(
                service_utils.translate_images,
                video_data.video.source_frames,
                annotations,
                output_dir=os.path.join(self.temp_dir, video_data.video_name),
                font_path="arial.ttf",
            )
            if not resp.status:
                raise Exception(resp.error)

            # === assembling_video ===
            yield self._progress_message("assembling_video", trace_id=trace_id)
            images_dir = os.path.join(self.temp_dir, video_data.video_name, 'frames_translated')
            new_audio_path = os.path.join(
                self.temp_dir, video_data.video_name, f'{self.audio_translate_name}.mp3'
            )
            output_video_path = os.path.join(self.temp_dir, f'{video_data.video_name}.mp4')

            resp = await self._run_blocking(
                service_utils.create_video_with_new_audio,
                images_dir=images_dir,
                original_video_path=video_data.source_path,
                new_audio_path=new_audio_path,
                output_video_path=output_video_path,
            )
            if not resp.status:
                raise Exception(resp.error)

            logger.info("[trace=%s] ml.pipeline.complete output=%s", trace_id, output_video_path)
            yield build_success(result={"output": f"{name}.mp4"})

        except Exception as e:
            logger.exception("[trace=%s] ❌ Ошибка в execute_stream", trace_id)
            yield build_error(
                code="ML_PROCESSING_FAILED",
                message=str(e),
                stage_failed="ml_processing",
            )
        finally:
            if dir_path and os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logger.info("[trace=%s] ml.cleanup.done workspace=%s", trace_id, dir_path)
                except Exception:
                    logger.warning(
                        "[trace=%s] Failed to cleanup ML workspace: %s",
                        trace_id, dir_path, exc_info=True,
                    )

    @staticmethod
    def _extract_trace_id(payload: dict) -> str:
        trace_meta = payload.get("_trace")
        if isinstance(trace_meta, dict):
            trace_id = trace_meta.get("trace_id")
            if trace_id:
                return str(trace_id)
        return "unknown"

    @staticmethod
    def _message_lag_ms(message: dict) -> int | None:
        timestamp = message.get("timestamp")
        if not timestamp:
            return None
        try:
            event_time = datetime.fromisoformat(str(timestamp))
        except ValueError:
            return None
        return int((datetime.now(timezone.utc) - event_time).total_seconds() * 1000)

    def _log_message_created(self, trace_id: str, message: dict) -> None:
        logger.debug(
            "[trace=%s] message_created stage=%s progress=%s status=%s msg_ts=%s details=%s",
            trace_id,
            message.get("stage"),
            message.get("progress"),
            message.get("status"),
            message.get("timestamp"),
            message.get("details"),
        )

    def _log_yield_boundary(
        self,
        trace_id: str,
        boundary: str,
        message: dict,
        *,
        resume_gap_ms: int | None = None,
    ) -> None:
        lag_ms = self._message_lag_ms(message)
        level = logging.DEBUG
        if (lag_ms is not None and lag_ms >= STREAM_LAG_WARNING_MS) or (
            resume_gap_ms is not None and resume_gap_ms >= STREAM_LAG_WARNING_MS
        ):
            level = logging.WARNING
        logger.log(
            level,
            "[trace=%s] %s stage=%s progress=%s status=%s msg_ts=%s lag_ms=%s resume_gap_ms=%s details=%s",
            trace_id,
            boundary,
            message.get("stage"),
            message.get("progress"),
            message.get("status"),
            message.get("timestamp"),
            lag_ms,
            resume_gap_ms,
            message.get("details"),
        )

    def __process_video(self, video_data: service_utils.VideoData) -> dict:
        """
        Проводит весь цикл обработки видео: извлечение, преобразование и сборка результата.

        Этапы:
            1. Разбиение видео на кадры.
            2. Извлечение аудио.
            3. Распознавание речи.
            4. Перевод текста.
            5. Генерация переведённого аудио.
            6. Обработка кадров.
            7. Переименование исходного видео.
            8. Сборка нового видео с синхронизацией аудио.

        Args:
            video_data (service_utils.VideoData): Объект с данными видео.

        Returns:
            dict: Словарь с результатом выполнения:
                - "status" (bool): Успешность операции.
                - "error" (str, optional): Сообщение об ошибке (если есть).
        """
        with log_duration(f'Обработка видео {video_data.video_name}'):
            video_data: service_utils.VideoData = service_utils.extract_frames(video_data)  
            video_data: service_utils.VideoData = self.frame_extractor.process(video_data)
        
        with log_duration("⏳Обработка"):
            video_data: service_utils.VideoData = self._audio_process(video_data)

            video_data: service_utils.VideoData = self._video_process(video_data)

        with log_duration("⏳Постобработка"):
            resp:service_utils.Response = service_utils.create_video_with_new_audio(video_data)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 
            
        # Удалим временные файлы и папки, если нужно
        temp_directory = os.path.join(self.temp_dir, video_data.video_name)
        service_utils.delete_temp_directory(temp_directory)
            
        return service_utils.Response(True, None, None)
    
    def _audio_process(self, video_data: service_utils.VideoData):
        with log_duration("Recognizer.process"):
            video_data: service_utils.VideoData = self.recognizer.process(video_data)
            
        with log_duration("Translator.translate"):
            video_data: service_utils.VideoData = self.translator.process(video_data)

        with log_duration("TextToSpeech.synthesize"):
            video_data: service_utils.VideoData = self.generator.process(video_data)
 
        return video_data

    def _video_process(self, video_data: service_utils.VideoData) -> service_utils.VideoData:
        with log_duration(message="OCR"):
            video_data: service_utils.VideoData = self.ocr.process(video_data)

        service_utils.save_video_data(video_data, 'var/video_data_example.pkl')

        with log_duration('Translate - API call'):
            video_data: service_utils.VideoData = self.translator.process(video_data)

        with log_duration('Re translate'):
            resp: service_utils.Response = service_utils.translate_images(
                video_data=video_data,
                output_dir=os.path.join(self.temp_dir, video_data.video_name),
                font_path="arial.ttf"
            )

        return resp
