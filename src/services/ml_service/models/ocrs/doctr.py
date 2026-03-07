from typing import List, Union, Dict, Any
from PIL import Image
import numpy as np
import torch
import json
from tqdm import tqdm

from doctr.io import DocumentFile, Document
from doctr.models import ocr_predictor
from src.services.ml_service.models.ocrs.base import BaseOCR


def ocr_to_dict(result: Document) -> List[Dict[str, Union[str, List[float]]]]:
    data = []
    for page in tqdm(result.pages, desc="OCR parsing pages", unit="page", ncols=100, colour="cyan"):
        page_data = []
        for block in page.blocks:
            for line in block.lines:
                text = ' '.join(word.value for word in line.words)
                (x_min, y_min), (x_max, y_max) = line.geometry
                page_data.append({
                    "text": text,
                    "bbox": [x_min, y_min, x_max, y_max]
                })
        data.append(page_data)
    return data


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

    def process(self, images: List[Union[str, Image.Image, np.ndarray]]):
        """Обработка одного изображения в документ."""
        try:
            doc = DocumentFile.from_images(images)
            result = self.model(doc)
            return ocr_to_dict(result)
        except Exception as e:
            return str(e)
        