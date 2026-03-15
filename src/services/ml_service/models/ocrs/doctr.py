from typing import List, Union, Dict, Any
from PIL import Image
import numpy as np
import torch
import json
from tqdm import tqdm
import logging

from doctr.io import DocumentFile, Document
from doctr.models import ocr_predictor
from src.services.ml_service.models.ocrs.base import BaseOCR
from ...utils import VideoData, Response, Video, Frame, TextItem, BoundingBox

logger = logging.getLogger(__name__)


def ocr_to_dict(result: Document, video_data: VideoData) -> VideoData:
    frames = []
    for page in tqdm(result.pages, desc="OCR parsing pages", unit="page", ncols=100, colour="cyan"):
        text_items = []
        for block in page.blocks:
            for line in block.lines:
                text = ' '.join(word.value for word in line.words)
                (x_min, y_min), (x_max, y_max) = line.geometry
                text_items.append(TextItem(
                    text=text,
                    bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                ))
        frames.append(Frame(texts=text_items))
    video_data.video.ocr_frames = frames
    return video_data


class DOCTR(BaseOCR):
    def __init__(self,
                 det_name: str = 'fast_base',
                 reco_name: str = 'crnn_vgg16_bn',
                 device: str = None):
        super().__init__(device=device, name='doctr')
        self.model = ocr_predictor(det_arch=det_name,
                                   reco_arch=reco_name,
                                   pretrained=True).to(self.device)
        self.model.eval()  # отключаем обучение

    def process(self, video_data: VideoData) -> VideoData:
        """Обработка одного изображения в документ."""
        try:
            images = [video_data.video.source_frames[idx] for idx in video_data.video.selected_frames]
            doc = DocumentFile.from_images(images)
            result = self.model(doc)
            return ocr_to_dict(result, video_data)
        except Exception as e:
            logger.error(f"OCR:{self.name} processing error: {e}")  
            return video_data
        