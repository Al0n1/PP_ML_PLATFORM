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

from . import utils as service_utils
from . import n_models as models
from src.services.ml_service.translator import Translator
from src.config.services.ml_config import settings
from src.utils.sse_messages import build_error, build_progress, build_success

STREAM_LAG_WARNING_MS = 5_000

@contextmanager
def log_duration(message: str | None = None, func_name: str | None = None):

    name = message or func_name

    # logger.info(f"⌛ {name}")
    start = perf_counter()

    yield

    end = perf_counter()
    logger.info(f"⏱ {name} finished in {end - start:.4f} sec")



logger = logging.getLogger(__name__)

STAGE_PROGRESS = {
    "copying_file": 5,
    "splitting_frames": 10,
    "extracting_audio": 25,
    "recognizing_speech": 40,
    "translating_text": 55,
    "generating_tts": 70,
    "processing_frames": 85,
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

    def _init_models(self):
        from huggingface_hub import login
        login()
        with log_duration("Translator.__init__"):
            self.translator = Translator(self.settings)
            # self.translator = models.UniversalTranslator(settings.TRANSLATOR_NAME, device=settings.TRANSLATOR_DEVICE, model_type=settings.TRANSLATOR_TYPE)
        
        with log_duration("SimpleWhisper.__init__"):
            self.recognizer = models.SimpleWhisper(device=self.settings.RECOGNIZER_DEVICE, model_name=self.settings.RECOGNIZER_NAME)

        with log_duration("TextToSpeech.__init__"):
            self.generator = models.TextToSpeech()

        with log_duration("OCR.__init__"):
            self.ocr = models.OCR(device=self.settings.OCR_DEVICE)

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

        if not path:
            logger.info("Video path is missing")
            return {"status": "error", "message": "Path missing"}

        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dir_path = os.path.join(self.temp_dir, name)
        os.makedirs(dir_path, exist_ok=True)

        # Запуск пайплайна обработки видео
        with log_duration(f'Обработка видео {name}'):
            resp: service_utils.Response = self.__process_video(path, name, dir_path)

        if resp.status is False:
            return {"status": "error", "message": str(resp.error)}

        result = {"status": "success", "result": {"output": f"{name}.mp4"}}

        logger.info(f"MLService.execute returning: {result}")
        return result

    async def execute_stream(self, data: dict) -> AsyncIterator[dict]:
        """
        Streaming версия execute() для SSE.
        """
        dir_path = None
        payload = data.get("data", data)
        trace_id = self._extract_trace_id(payload)
        try:
            path = payload.get("path", "")
            if not path:
                error_message = build_error(
                    code="ML_PROCESSING_FAILED",
                    message="Path missing",
                    stage_failed="validation",
                )
                self._log_message_created(trace_id, error_message)
                self._log_yield_boundary(trace_id, "before_stream_yield", error_message)
                yield_started = perf_counter()
                yield error_message
                self._log_yield_boundary(
                    trace_id,
                    "after_stream_resume",
                    error_message,
                    resume_gap_ms=int((perf_counter() - yield_started) * 1000),
                )
                return

            filename = os.path.basename(path)
            name = os.path.splitext(filename)[0]
            dir_path = os.path.join(self.temp_dir, name)
            os.makedirs(dir_path, exist_ok=True)

            logger.info(
                "[trace=%s] ml.execute_stream.start input_path=%s workspace=%s",
                trace_id,
                path,
                dir_path,
            )

            # Основной streaming-пайплайн
            async for msg in self.__process_video_stream(path, name, dir_path, trace_id=trace_id):
                self._log_yield_boundary(trace_id, "before_stream_yield", msg)
                yield_started = perf_counter()
                yield msg
                self._log_yield_boundary(
                    trace_id,
                    "after_stream_resume",
                    msg,
                    resume_gap_ms=int((perf_counter() - yield_started) * 1000),
                )

            # Успешное завершение
            success_message = build_success(
                result={"output": f"{name}.mp4"}
            )
            self._log_message_created(trace_id, success_message)
            self._log_yield_boundary(trace_id, "before_stream_yield", success_message)
            yield_started = perf_counter()
            yield success_message
            self._log_yield_boundary(
                trace_id,
                "after_stream_resume",
                success_message,
                resume_gap_ms=int((perf_counter() - yield_started) * 1000),
            )

        except Exception as e:
            logger.exception("[trace=%s] ❌ Ошибка в execute_stream", trace_id)
            error_message = build_error(
                code="ML_PROCESSING_FAILED",
                message=str(e),
                stage_failed="ml_processing",
            )
            self._log_message_created(trace_id, error_message)
            self._log_yield_boundary(trace_id, "before_stream_yield", error_message)
            yield_started = perf_counter()
            yield error_message
            self._log_yield_boundary(
                trace_id,
                "after_stream_resume",
                error_message,
                resume_gap_ms=int((perf_counter() - yield_started) * 1000),
            )
        finally:
            if dir_path and os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logger.info("[trace=%s] ml.cleanup.done workspace=%s", trace_id, dir_path)
                except Exception:
                    logger.warning(
                        "[trace=%s] Failed to cleanup ML workspace: %s",
                        trace_id,
                        dir_path,
                        exc_info=True,
                    )

    async def __process_video_stream(self, path: str, name: str, result_dir: str, trace_id: str):
        """
        Streaming версия обработки видео с прогрессом.
        """

        base_dir = os.path.join(self.temp_dir, name)
        extract_audio_path = os.path.join(base_dir, f'{self.audio_extract_name}.mp3')
        frames_output_dir = os.path.join(base_dir, 'frames')
        logger.info(
            "[trace=%s] ml.pipeline.start base_dir=%s source_path=%s",
            trace_id,
            base_dir,
            path,
        )

        # === ЭТАП 1: copying_file ===
        yield self._progress_message("copying_file", trace_id=trace_id)
        stage_started = perf_counter()
        await self._run_blocking(shutil.copy, path, base_dir)
        logger.info(
            "[trace=%s] ml.stage.done stage=copying_file duration_ms=%s destination=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            base_dir,
        )

        # === ЭТАП 2: splitting_frames ===
        yield self._progress_message("splitting_frames", trace_id=trace_id)
        stage_started = perf_counter()
        r = await self._run_blocking(service_utils.extract_frames, path, frames_output_dir)
        if not r.status:
            raise Exception(r.error)

        images = await self._run_blocking(service_utils.get_image_paths, frames_output_dir)
        logger.info(
            "[trace=%s] ml.stage.done stage=splitting_frames duration_ms=%s frames=%s output_dir=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            len(images),
            frames_output_dir,
        )

        # === ЭТАП 3: extracting_audio ===
        yield self._progress_message("extracting_audio", trace_id=trace_id)
        stage_started = perf_counter()
        r = await self._run_blocking(service_utils.extract_audio, path, extract_audio_path)
        if not r.status:
            raise Exception(r.error)
        logger.info(
            "[trace=%s] ml.stage.done stage=extracting_audio duration_ms=%s audio_path=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            extract_audio_path,
        )

        # === ЭТАП 4: recognizing_speech ===
        yield self._progress_message("recognizing_speech", trace_id=trace_id)
        stage_started = perf_counter()
        r = await self._run_blocking(self.recognizer.transcribe, extract_audio_path)
        if not r.status:
            raise Exception(r.error)
        transcript = r.result
        logger.info(
            "[trace=%s] ml.stage.done stage=recognizing_speech duration_ms=%s transcript_chars=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            len(transcript or ""),
        )

        # === ЭТАП 5: translating_text ===
        yield self._progress_message("translating_text", trace_id=trace_id)
        stage_started = perf_counter()
        r = await self._run_blocking(self.translator.translate, transcript)
        if not r.status:
            raise Exception(r.error)
        translation = r.result
        logger.info(
            "[trace=%s] ml.stage.done stage=translating_text duration_ms=%s translation_chars=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            len(translation or ""),
        )

        # === ЭТАП 6: generating_tts ===
        yield self._progress_message("generating_tts", trace_id=trace_id)
        wav_path = os.path.join(base_dir, f'{self.audio_translate_name}.wav')
        mp3_path = os.path.join(base_dir, f'{self.audio_translate_name}.mp3')

        stage_started = perf_counter()
        r = await self._run_blocking(self.generator.synthesize, translation, output_path=wav_path)
        if not r.status:
            raise Exception(r.error)

        r = await self._run_blocking(service_utils.wav_to_mp3, wav_path, mp3_path)
        if not r.status:
            raise Exception(r.error)
        logger.info(
            "[trace=%s] ml.stage.done stage=generating_tts duration_ms=%s wav_path=%s mp3_path=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            wav_path,
            mp3_path,
        )

        # === ЭТАП 7: processing_frames (С ПОДЭТАПАМИ) ===
        total_images = len(images)
        yield self._progress_message("processing_frames", trace_id=trace_id)
        logger.info("[trace=%s] ml.stage.start stage=processing_frames total_images=%s", trace_id, total_images)

        stage_started = perf_counter()
        ocr_response = await self._run_blocking(self.ocr.batch, images)
        if not ocr_response.status:
            raise Exception(ocr_response.error)
        ocr_raw = ocr_response.result
        ocr_results = await self._run_blocking(self.ocr.ocr_to_dict, ocr_raw)
        translated_response = await self._run_blocking(
            service_utils.translate_ocr_results,
            self.translator, ocr_results
        )
        if not translated_response.status:
            raise Exception(translated_response.error)
        translated = translated_response.result

        out_dir = os.path.join(base_dir, "frames_translated")
        os.makedirs(out_dir, exist_ok=True)
        logger.info(
            "[trace=%s] ml.processing_frames.prepare_done duration_ms=%s ocr_items=%s translated_items=%s output_dir=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            len(ocr_results),
            len(translated),
            out_dir,
        )

        for index, img in enumerate(images, start=1):
            image_started = perf_counter()
            image_response = await self._run_blocking(
                service_utils.translate_images,
                [img],
                translated[index - 1:index],
                output_dir=out_dir,
                font_path="arial.ttf",
            )
            if not image_response.status:
                raise Exception(image_response.error)
            yield self._progress_message(
                "processing_frames",
                current_step=index,
                total_steps=total_images,
                trace_id=trace_id,
            )
            if index == 1 or index == total_images or index % 25 == 0:
                logger.info(
                    "[trace=%s] ml.processing_frames.step index=%s total=%s duration_ms=%s image=%s",
                    trace_id,
                    index,
                    total_images,
                    int((perf_counter() - image_started) * 1000),
                    img,
                )

        # === ЭТАП 8: assembling_video ===
        yield self._progress_message("assembling_video", trace_id=trace_id)
        stage_started = perf_counter()
        r = await self._run_blocking(
            service_utils.create_video_with_new_audio,
            images_dir=out_dir,
            original_video_path=path,
            new_audio_path=mp3_path,
            output_video_path=os.path.join(self.temp_dir, f"{name}.mp4"),
        )
        if not r.status:
            raise Exception(r.error)
        logger.info(
            "[trace=%s] ml.stage.done stage=assembling_video duration_ms=%s output=%s",
            trace_id,
            int((perf_counter() - stage_started) * 1000),
            os.path.join(self.temp_dir, f"{name}.mp4"),
        )
        logger.info("[trace=%s] ml.pipeline.complete output=%s", trace_id, os.path.join(self.temp_dir, f"{name}.mp4"))

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


    def __process_video(self, path: str, name: str, result_dir: str) -> dict:
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
            path (str): Путь к исходному видеофайлу.
            name (str): Имя видео (используется как префикс временных файлов).
            result_dir (str): Путь к директории для сохранения результатов.

        Returns:
            dict: Словарь с результатом выполнения:
                - "status" (bool): Успешность операции.
                - "error" (str, optional): Сообщение об ошибке (если есть).
        """
        # Основная логика обработки файла

        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        dir_path = os.path.join(self.temp_dir, name)
        os.makedirs(dir_path, exist_ok=True)

        extract_audio_path = os.path.join(self.temp_dir, name, f'{self.audio_extract_name}.mp3')
        frames_output_dir = os.path.join(self.temp_dir, name, 'frames')

        with log_duration("Предобработка"):
            resp: service_utils.Response = service_utils.extract_audio(path, extract_audio_path)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 
            
            # resp: service_utils.Response = service_utils.extract_frames(path, frames_output_dir)  
            key_frames, total_frames = service_utils.extract_key_frames(path, frames_output_dir, save_all_frames=True)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)          

        with log_duration("Обработка"):
            resp: service_utils.Response = self._audio_process(path, self.temp_dir, name)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 
            
            resp: service_utils.Response = self._video_process(path, self.temp_dir, name, {'key_frames': key_frames, 'total_frames': total_frames})
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 

        images_dir=os.path.join(self.temp_dir, name, 'frames_translated')
        original_video_path = path
        new_audio_path = os.path.join(self.temp_dir, name, f'{self.audio_translate_name}.mp3')
        output_video_path=os.path.join(self.temp_dir, f'{name}.mp4')

        with log_duration("Постобработка"):
            resp:service_utils.Response = service_utils.create_video_with_new_audio(    
                images_dir=images_dir,
                original_video_path=original_video_path,
                new_audio_path=new_audio_path,
                output_video_path=output_video_path
            )
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 
        try:
            shutil.rmtree(dir_path)
            logger.info(f"✅Директория '{dir_path}' успешно удалена.")
        except FileNotFoundError:
            logger.info(f"❌Директория '{dir_path}' не найдена.")
        except Exception as e:
            logger.info(f"❌Ошибка при удалении директории: {e}")
        
        return service_utils.Response(True, None, None)
    
    def _audio_process(self, path, temp_dir, name):
        base_dir = os.path.join(temp_dir, name)
        extract_audio_path = os.path.join(base_dir, f"{self.audio_extract_name}.mp3")
        translated_wav = os.path.join(base_dir, f"{self.audio_translate_name}.wav")
        translated_mp3 = os.path.join(base_dir, f"{self.audio_translate_name}.mp3")

        with log_duration("SimpleWhisper.transcribe"):
            resp: service_utils.Response = self.recognizer.transcribe(extract_audio_path)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
            transcript = resp.result
            path = os.path.join(temp_dir, name, f"audio_text_transcript.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
        with log_duration("Translator.translate"):
            resp: service_utils.Response = self.translator.translate([transcript])
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
            translation = resp.result['text'][0] # берем первый элемент, так как передавали список из одного текста
            path = os.path.join(temp_dir, name, f"audio_text_translation.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(translation) # берем первый элемент, так как передавали список из одного текста

        with log_duration("TextToSpeech.synthesize"):
            resp: service_utils.Response = self.generator.synthesize(translation, output_path=translated_wav)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)

        with log_duration("wav_to_mp3"):
            resp: service_utils.Response = service_utils.wav_to_mp3(translated_wav, translated_mp3)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
        
        return service_utils.Response(True, None, None)

    def _video_process(self, path, temp_dir, name, extra_info):
        log_duration("Video OCR and Translation")
        key_frames = extra_info.get('key_frames', [])
        total_frames = extra_info.get('total_frames', 0)

        frames_dir = os.path.join(temp_dir, name, 'frames')
        images = service_utils.get_image_paths(frames_dir)
        output_dir = os.path.join(temp_dir, name, 'frames_translated')
        ocr_out_path = os.path.join(temp_dir, name, f'ocr.json')

        chosed_images = [img for idx, img in enumerate(images) if idx in key_frames]


        with log_duration("OCR"):
            resp: service_utils.Response = self.ocr.batch(chosed_images)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
            results = resp.result

            resp: service_utils.Response = self.ocr.save_results_to_json(results, ocr_out_path)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
            
        from tqdm import tqdm
        with log_duration("Translate"):
            results = service_utils.load_json(ocr_out_path)
            for i, result in enumerate(
                tqdm(results, desc="Translating OCR texts", unit="frame", ncols=100, colour="green"),
                start=1,
            ):
                texts = [item['text'] for item in result]
                # logger.info(f"[frame {i}/{len(results)}] Translating {len(texts)} text(s)")
                resp: service_utils.Response = self.translator.translate(texts)
                # print(resp.result)
                for item, translate in zip(result, resp.result['text']):
                    item['translation'] = translate
            resp: service_utils.Response = service_utils.Response(True, None, results)
            # resp: service_utils.Response = service_utils.translate_ocr_results(self.translator, results)
            if resp.status is False:
                return service_utils.Response(False, resp.error, None)
            translated_data = resp.result
            service_utils.save_json(translated_data, os.path.join(temp_dir, name, 'video_text.json'))

        id_to_id = {k: i for i, k in enumerate(key_frames)}
        print(type(translated_data))
        indices = key_frames
        t = total_frames

        results = []
        indices_ext = key_frames + [total_frames]

        results = [
            k
            for i, k in enumerate(key_frames)
            for _ in range(indices_ext[i+1] - k)
        ]

        print(results, len(results))

        imgs = [images[id_result] for id_result in results]
        trans = [translated_data[id_to_id[id_result]] for id_result in results]
        with log_duration('Re translate'):
            resp: service_utils.Response = service_utils.translate_images(
                imgs,
                trans,
                output_dir=output_dir,
                font_path="arial.ttf"
            )
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 





        # with log_duration('Re translate'):
        #     resp: service_utils.Response = service_utils.translate_images(
        #         chosed_images,
        #         translated_data,
        #         output_dir=output_dir,
        #         font_path="arial.ttf"
        #     )
            if resp.status is False:
                return service_utils.Response(False, resp.error, None) 
        
        return service_utils.Response(True, None, None)
