from typing import Union, Any

from src.config.services.ml_config import settings, Settings
from src.services.ml_service.models.recognizers.base import BaseRecognizer
from ..utils import Response 

MODELS = {
    'whisper': ['tiny', 'base', 'small', 'medium', 'large']
}

class Recognizer:
    def __init__(self, settings: Settings = settings):
        self.settings: Settings = settings

        if self.settings.RECOGNIZER_TYPE == 'whisper':
            if self.settings.RECOGNIZER_NAME not in MODELS['whisper']:
                raise ValueError(f"Unsupported Whisper model: {self.settings.RECOGNIZER_NAME}")
            from src.services.ml_service.models.recognizers.openai_whisper import WhisperSpeechRecognitionModel
            self.model = WhisperSpeechRecognitionModel(model_name=self.settings.RECOGNIZER_NAME, cache_dir=self.settings.MODEL_CACHE_DIR)
        
    
    def process(self, data: Union[str, Any]) -> Response:
        result = self.model.process(data)
        if isinstance(result, str):  # если результат - строка, значит произошла ошибка
            return Response(False, result, None)
        return Response(True, None, result)
