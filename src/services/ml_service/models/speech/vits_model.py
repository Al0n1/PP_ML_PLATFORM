from ...utils import VideoData
from .base import BaseSpeechModel
import logging
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy.io.wavfile as wvfile
from pydub import AudioSegment
from ...utils import wav_to_mp3
from src.config.services.ml_config import settings, Settings

logger = logging.getLogger(__name__)


class VitsAudioGenerationModel(BaseSpeechModel):
    def __init__(self, model_name="facebook/mms-tts-rus", cache_dir=settings.MODEL_CACHE_DIR, temp_dir=settings.TEMP_DIR):
        self.model_name = model_name
        self.model = VitsModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.temp_dir = temp_dir
        logger.info(f"Loaded model and tokenizer: {model_name}")

    def process(self, video_data: VideoData) -> VideoData:
        inputs = self.tokenizer(video_data.audio.translated_text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**inputs).waveform
            
        # Convert tensor to numpy and scale to int16
        waveform = output[0].cpu().numpy()
        waveform_int16 = (waveform * 32767).astype(np.int16)
            
        wav_path = f"{self.temp_dir}/translated_{video_data.video_name}.wav"
        mp3_path = f"{self.temp_dir}/translated_{video_data.video_name}.mp3"

        video_data.audio.output_audio_path = wav_path
            
        # Write WAV file
        wvfile.write(wav_path, rate=self.model.config.sampling_rate, data=waveform_int16)
        logger.info(f"WAV file saved to: {wav_path}")
        # Convert WAV to MP3
        wav_to_mp3(wav_path, mp3_path)
        logger.info(f"MP3 file saved to: {mp3_path}")
        video_data.audio.output_audio_path = mp3_path
        return video_data
