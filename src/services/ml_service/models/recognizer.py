from typing import Union, Any

from src.config.services.ml_config import settings, Settings
from src.services.ml_service.models.recognizers.base import BaseRecognizer
from .recognizers.openai_whisper import WhisperSpeechRecognitionModel, whisper
from ..utils import Response, VideoData

MODELS = {
    'whisper': whisper.available_models(),
}

class Recognizer(BaseRecognizer):
    def __init__(self, settings: Settings = settings):
        super().__init__(name="Recognizer")
        self.settings: Settings = settings

        if self.settings.RECOGNIZER_TYPE == 'whisper':
            if self.settings.RECOGNIZER_NAME not in MODELS['whisper']:
                raise ValueError(f"Unsupported Whisper model: {self.settings.RECOGNIZER_NAME}")
            self.model = WhisperSpeechRecognitionModel(model_name=self.settings.RECOGNIZER_NAME, cache_dir=self.settings.MODEL_CACHE_DIR)
        
    
    def process(self, video_data: VideoData) -> VideoData:
        video_data = self.model.process(video_data)
        return video_data
