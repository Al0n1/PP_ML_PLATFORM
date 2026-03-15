from typing import List, Any, Union
from ...utils import Response, VideoData

class BaseRecognizer:
    def __init__(self, name: str = 'base_recognizer'):
        self.name = name

    def process(self, video_data: VideoData) -> VideoData:
        raise NotImplementedError("Subclasses must implement this method")
    