from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar

load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # ===== Paths =====
    TEMP_DIR: str = "var/temp"
    MODEL_CACHE_DIR: str = "var/model_cache"
    ML_CLI_SOURCE_PATH: str = ""
    FFMPEG_BINARY: str = ""
    FFPROBE_BINARY: str = ""

    # ===== Translator =====
    TRANSLATOR_NAME: str = "glazzova/translation_en_ru"
    TRANSLATOR_TYPE: str = "marian"
    TRANSLATOR_DEVICE: str = "mps"

    # ===== Speech recognition =====
    RECOGNIZER_NAME: str = "medium"
    RECOGNIZER_DEVICE: str = "cpu"

    # ===== OCR =====
    OCR_TYPE: str = "doctr"
    OCR_NAME: str = "fast_base|||crnn_vgg16_bn"
    OCR_DEVICE: str = "mps"


settings = Settings()
