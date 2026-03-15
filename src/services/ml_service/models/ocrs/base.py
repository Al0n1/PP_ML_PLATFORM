from typing import List, Dict, Union
from PIL import Image
import numpy as np
import json
from ...utils import VideoData, Response, Video, Frame, TextItem, BoundingBox


class BaseOCR:
    def __init__(self, device: str = None, name: str = 'base'):
        self.device = device
        self.name = name

    def process(self, video_data: VideoData) -> VideoData:
        raise NotImplementedError("Subclasses must implement this method")

    def save(self, video_data: VideoData, output_path: str) -> bool | str:
        if video_data.video.ocr_frames is None:
            return False

        result = []
        for frame_idx, frame in enumerate(video_data.video.ocr_frames):
            frame_data = {"frame_index": frame_idx, "texts": []}
            if frame.texts:
                for item in frame.texts:
                    entry = {
                        "text": item.text,
                        "translation": item.translation,
                    }
                    if item.bounding_box:
                        entry["bbox"] = [
                            item.bounding_box.x_min,
                            item.bounding_box.y_min,
                            item.bounding_box.x_max,
                            item.bounding_box.y_max,
                        ]
                    frame_data["texts"].append(entry)
            result.append(frame_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return output_path
    