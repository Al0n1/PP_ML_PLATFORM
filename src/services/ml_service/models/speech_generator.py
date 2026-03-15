from ....config.services.ml_config import settings, Settings
from ..utils import VideoData
from .speech.vits_model import VitsAudioGenerationModel


class SpeechModel:
    def __init__(self, config: Settings):
        if config.GENERATOR_TYPE == "vits":
            
            self.model = VitsAudioGenerationModel(
                model_name=config.GENERATOR_NAME,
                cache_dir=config.MODEL_CACHE_DIR,
                temp_dir=config.TEMP_DIR,
            )
        else:
            raise ValueError(f"Unsupported GENERATOR_TYPE: {config.GENERATOR_TYPE}")
  

    def process(self, video_data: VideoData) -> VideoData:
        return self.model.process(video_data)