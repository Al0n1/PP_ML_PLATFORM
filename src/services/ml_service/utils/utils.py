import pickle
import shutil 
import logging
import numpy as np
from typing import Optional, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def delete_temp_directory(temp_directory_path: str):
    try:
        shutil.rmtree(temp_directory_path)
        logger.info(f"✅Директория '{temp_directory_path}' успешно удалена.")
    except FileNotFoundError:
        logger.info(f"❌Директория '{temp_directory_path}' не найдена.")
    except Exception as e:
        logger.info(f"❌Ошибка при удалении директории: {e}")    



class Response:
    def __init__(self, status: bool, error: str|None, result: Any):
        self.status = status
        self.error = error
        self.result = result

@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass
class TextItem:
    text: Optional[str] = None
    translation: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None

@dataclass
class Frame:
    texts: Optional[List[TextItem]] = None

    bounding_boxes: Optional[List[BoundingBox]] = None
    source_texts: Optional[List[str]] = None
    translated_texts: Optional[List[str]] = None

@dataclass
class Video:
    source_frames: Optional[List[np.ndarray]] = None
    selected_frames: Optional[List[int]] = None
    ocr_frames: Optional[List[Frame]] = None
    translated_frames: bool = False
    translated_frames_indexes: Optional[List[int]] = None

@dataclass
class Audio:
    source_audio_path: Optional[str] = None
    source_text: Optional[str] = None
    translated_text: Optional[str] = None
    output_audio_path: Optional[str] = None

@dataclass
class VideoData:
    # Base
    source_path: Optional[str] = None
    video_name: Optional[str] = None
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    output_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    # Audio
    audio: Audio = field(default_factory=Audio)
    # Video
    video: Video = field(default_factory=Video)


def save_video_data(video_data: 'VideoData', path: str):
    try:
        with open(path, 'wb') as f:
            pickle.dump(video_data, f)
        logger.info(f"✅VideoData успешно сохранены в '{path}'.")
    except Exception as e:
        logger.error(f"❌Ошибка при сохранении VideoData: {e}")

def load_video_data(path: str) -> VideoData:
    with open(path, 'rb') as f:
        video_data = pickle.load(f)
    return video_data
        


if __name__ == '__main__':
    vd = VideoData()
    

    
    
