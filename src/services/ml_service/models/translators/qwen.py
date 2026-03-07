"""Реализация модели перевода текста на базе Qwen2."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List

from .base import BaseTranslationModel
from src.config.services.ml_config import settings

logger = logging.getLogger(__name__)

TRANSLATION_PROMPT = (
    "Translate the following text from English to Russian.\n"
    "Output ONLY the translation, nothing else.\n\n"
    "Text: {text}\n"
    "Translation:"
)


class QwenTextTranslationModel(BaseTranslationModel):
    """Модель перевода на основе Qwen2-2B (авторегрессионная генерация).

    Использует промпт-инструкцию для перевода текста EN → RU.

    Args:
        model_checkpoint: Идентификатор модели Hugging Face.
        cache_dir: Локальный каталог для кеширования модели и токенизатора.
        max_new_tokens: Максимальное количество генерируемых токенов.

    Raises:
        Exception: Любая ошибка при загрузке токенизатора или модели.
    """

    def __init__(
        self,
        model_checkpoint: str = "Qwen/Qwen2-1.5B-Instruct",
        cache_dir: str = settings.MODEL_CACHE_DIR,
        max_new_tokens: int = 512,
    ):
        super().__init__(image_support=False)
        self.max_new_tokens = max_new_tokens

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            logger.info("Qwen model and tokenizer loaded successfully")

            self.device = torch.device(
                settings.TRANSLATOR_DEVICE if torch.cuda.is_available() else "cpu"
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading Qwen model/tokenizer: {e}")
            raise e

    def _build_prompt(self, text: str) -> str:
        """Формирует промпт для перевода одного фрагмента.

        Args:
            text: Исходный текст на английском языке.

        Returns:
            str: Готовый промпт для модели.
        """
        return TRANSLATION_PROMPT.format(text=text)

    @torch.inference_mode()
    def _generate(self, prompt: str) -> str:
        """Генерирует ответ модели по промпту.

        Args:
            prompt: Промпт для генерации.

        Returns:
            str: Сгенерированный текст (без промпта).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

        generated_ids = outputs[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def translate(self, texts: List[str]) -> dict:
        """Перевод списка текстов.

        Args:
            texts: Список исходных текстов для перевода.

        Returns:
            dict: Результат перевода с ключами:
                ``status`` (bool), ``text`` (list[str]) или ``error`` (str),
                и ``source_text`` (list[str]).
        """
        if not texts:
            return {"status": False, "error": "Empty input text", "source_text": texts}

        try:
            translated: List[str] = []
            for text in texts:
                prompt = self._build_prompt(text)
                result = self._generate(prompt)
                translated.append(result)
                logger.info(f"Translated: {text[:60]!r} -> {result[:60]!r}")

            return {"status": True, "text": translated, "source_text": texts}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"status": False, "error": str(e), "source_text": texts}

    def __call__(self, text: str) -> dict:
        """Перевод одного текста.

        Args:
            text: Исходный текст для перевода.

        Returns:
            dict: Результат перевода с ключами:
                ``status`` (bool), ``text`` (str) или ``error`` (str),
                и ``source_text`` (str).
        """
        if not text.strip():
            return {"status": False, "error": "Empty input text", "source_text": text}

        try:
            prompt = self._build_prompt(text)
            translated = self._generate(prompt)
            logger.info(f"Translated: {text[:60]!r} -> {translated[:60]!r}")
            return {"status": True, "text": translated, "source_text": text}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"status": False, "error": str(e), "source_text": text}
