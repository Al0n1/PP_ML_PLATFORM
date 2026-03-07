"""Реализация модели перевода текста на базе Marian."""

import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import List

from .base import BaseTranslationModel
from src.config.services.ml_config import settings

logger = logging.getLogger(__name__)

# Доля свободной VRAM, которую разрешено занимать под один батч (с запасом)
_GPU_MEM_USAGE_FRACTION = 0.8
# Приблизительный расход памяти на один токен при генерации (в байтах, fp16)
_BYTES_PER_TOKEN_FP16 = 512
_BYTES_PER_TOKEN_FP32 = 1024
# Минимальный / максимальный размер батча
_MIN_BATCH_SIZE = 1
_MAX_BATCH_SIZE = 128


class OpusTextTranslationModel(BaseTranslationModel):
    """Модель перевода на основе контрольных точек Helsinki-NLP Marian.

    Поддерживает:
      - Автоматический выбор FP16 при наличии совместимой GPU.
      - Динамическое определение размера батча на основе свободной VRAM.
      - Пакетную обработку произвольно длинных списков текстов.

    Args:
        model_checkpoint: Идентификатор модели Hugging Face.
        cache_dir: Локальный каталог для кеширования модели и токенизатора.
        gpu_mem_fraction: Доля свободной VRAM для одного батча (0..1).

    Raises:
        Exception: Любая ошибка при загрузке токенизатора или модели.
    """

    def __init__(
        self,
        model_checkpoint="Helsinki-NLP/opus-mt-en-ru",
        cache_dir=settings.MODEL_CACHE_DIR,
        gpu_mem_fraction: float = _GPU_MEM_USAGE_FRACTION,
    ):
        super().__init__(image_support=False)
        self._gpu_mem_fraction = gpu_mem_fraction

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)

            self.device = torch.device(settings.TRANSLATOR_DEVICE if torch.cuda.is_available() else "cpu")
            self._use_fp16 = self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 7

            if self._use_fp16:
                self.model.half()
                logger.info("Model converted to FP16 for GPU acceleration")

            self.model.to(self.device)
            self.model.eval()
            logger.info(
                "Model loaded | device=%s fp16=%s checkpoint=%s",
                self.device, self._use_fp16, model_checkpoint,
            )
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {e}")
            raise

    # ------------------------------------------------------------------
    # Определение размера батча
    # ------------------------------------------------------------------

    def _estimate_batch_size(self, max_seq_len: int) -> int:
        """Оценивает допустимый размер батча по свободной VRAM.

        Args:
            max_seq_len: Максимальная длина последовательности (в токенах) в текущем наборе.

        Returns:
            Рекомендуемый размер батча.
        """
        if self.device.type != "cuda":
            return _MAX_BATCH_SIZE

        torch.cuda.synchronize(self.device)
        free, _ = torch.cuda.mem_get_info(self.device)
        usable = free * self._gpu_mem_fraction

        bytes_per_token = _BYTES_PER_TOKEN_FP16 if self._use_fp16 else _BYTES_PER_TOKEN_FP32
        # encoder + decoder ≈ 2× seq_len, beam search ~4× overhead
        tokens_per_sample = max_seq_len * 2 * 4
        mem_per_sample = tokens_per_sample * bytes_per_token

        if mem_per_sample == 0:
            return _MAX_BATCH_SIZE

        batch_size = max(_MIN_BATCH_SIZE, min(_MAX_BATCH_SIZE, int(usable / mem_per_sample)))
        logger.debug(
            "Batch size estimation: free_vram=%.1fMB usable=%.1fMB seq_len=%d → batch_size=%d",
            free / 1e6, usable / 1e6, max_seq_len, batch_size,
        )
        return batch_size

    # ------------------------------------------------------------------
    # Пакетный перевод
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _translate_batch(self, texts: List[str]) -> List[str]:
        """Переводит один батч текстов (уже подобранный по размеру).

        Args:
            texts: Список текстов одного батча.

        Returns:
            Список переведённых строк.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if inputs["input_ids"].size(1) == 0:
            return [""] * len(texts)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs)

        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def translate(self, texts: List[str], image=None) -> dict:
        """Перевод списка текстов с автоматической пакетной обработкой.

        Тексты разбиваются на батчи, размер которых определяется динамически
        на основе свободной памяти GPU и длины входных последовательностей.

        Args:
            texts: Список исходных текстов для перевода.

        Returns:
            dict: Результат перевода с ключами:
                - ``status`` (bool)
                - ``text`` (list[str]) — переведённые тексты (при успехе)
                - ``error`` (str) — описание ошибки (при неудаче)
                - ``source_text`` (list[str])
        """
        if not texts:
            return {'status': False, 'error': 'Empty input text', 'source_text': texts}

        try:
            # Предварительная токенизация для оценки длины
            pre_encoded = self.tokenizer(texts, padding=False, truncation=True)
            max_seq_len = max(len(ids) for ids in pre_encoded["input_ids"]) if pre_encoded["input_ids"] else 1
            batch_size = self._estimate_batch_size(max_seq_len)

            total_batches = math.ceil(len(texts) / batch_size)
            logger.info(
                "Translating %d text(s) in %d batch(es) (batch_size=%d, max_seq_len=%d)",
                len(texts), total_batches, batch_size, max_seq_len,
            )

            translated_texts: List[str] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_num = i // batch_size + 1
                logger.debug("Batch %d/%d — %d text(s)", batch_num, total_batches, len(batch))

                translated_texts.extend(self._translate_batch(batch))

                # Освобождаем неиспользуемый кеш между батчами
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            return {'status': True, 'text': translated_texts, 'source_text': texts}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {'status': False, 'error': str(e), 'source_text': texts}

