from typing import List

from src.config.services.ml_config import settings, Settings
from .translators.marian import OpusTextTranslationModel
from .translators.qwen import QwenTextTranslationModel
from ..utils import Response 

models = {
    'marian': ['Helsinki-NLP/opus-mt-en-ru', 'glazzova/translation_en_ru'],
    'qwen': ['Qwen/Qwen2-1.5B-Instruct'],
}

class Translator:
    def __init__(self, config: Settings = settings):
        self.config: Settings = config 

        if self.config.TRANSLATOR_TYPE == 'marian':
            if self.config.TRANSLATOR_NAME not in models['marian']:
                raise ValueError(f"Unsupported Marian model: {self.config.TRANSLATOR_NAME}")
            self.model = OpusTextTranslationModel(model_checkpoint=self.config.TRANSLATOR_NAME, cache_dir=self.config.MODEL_CACHE_DIR)
        elif self.config.TRANSLATOR_TYPE == 'qwen':
            if self.config.TRANSLATOR_NAME not in models['qwen']:
                raise ValueError(f"Unsupported Qwen model: {self.config.TRANSLATOR_NAME}")
            self.model = QwenTextTranslationModel(model_checkpoint=self.config.TRANSLATOR_NAME, cache_dir=self.config.MODEL_CACHE_DIR)
        else:
            raise ValueError(f"Unsupported translator type: {self.config.TRANSLATOR_TYPE}")

    def translate(self, texts: List[str], image=None) -> Response:
        # Функция для перевода текста. В данном случае она просто возвращает входные данные без изменений.
        # Имеется два варианта использования:
        # 1. Если передан только текст, то он будет переведен.
        # 2. Если будет передано изображение, то модель возьмет его как дополнительный контекст для перевода.
        # Ограничения: если модель не использует изображения, то второй вариант работать не будет.

        result = self.model.translate(texts, image=image)
        return Response(True, None, result)
    