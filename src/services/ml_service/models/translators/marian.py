"""Реализация модели перевода текста на базе Marian."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import List

from .base import BaseTranslationModel
from src.config.services.ml_config import settings

logger = logging.getLogger(__name__)

class OpusTextTranslationModel(BaseTranslationModel):
    """Модель перевода на основе контрольных точек Helsinki-NLP Marian.

    Args:
        model_checkpoint: Идентификатор модели Hugging Face.
        cache_dir: Локальный каталог для кеширования модели и токенизатора.

    Raises:
        Exception: Любая ошибка при загрузке токенизатора или модели.
    """

    def __init__(self, model_checkpoint="Helsinki-NLP/opus-mt-en-ru", cache_dir=settings.MODEL_CACHE_DIR):
        super().__init__(image_support=False)            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir=cache_dir)
            logger.info("Model and tokenizer loaded successfully")

            self.device = torch.device(settings.TRANSLATOR_DEVICE if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Error loading model/tokenizer: {e}")
            raise e
    
    @torch.inference_mode()
    def translate(self, texts: List[str]):
        """Перевод списка текстов.

        Args:
            texts: Список исходных текстов для перевода.

        Returns:
            dict: Результат перевода с ключами: ``status`` (bool), ``text`` (list[str]) или ``error`` (str), и ``source_text`` (list[str]).
        """

        if not texts:
            return {'status': False, 'error': 'Empty input text', 'source_text': texts}
        
        try:
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            if inputs["input_ids"].size(1) == 0:
                return {'status': False, 'error': 'No tokens produced by tokenizer', 'source_text': texts}

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs)
            translated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            return {'status': True, 'text': translated_texts, 'source_text': texts}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {'status': False, 'error': str(e), 'source_text': texts}

