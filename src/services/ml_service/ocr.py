from typing import List, Union
import numpy as np
import cv2
from PIL import Image

from .utils import Response 
from src.config.services.ml_config import settings, Settings
from src.services.ml_service.models.ocrs.doctr import DOCTR 
from src.services.ml_service.models.ocrs.ocr_paddle import PaddleOCRModel


MODELS = {'doctr': ['fast_base|||crnn_vgg16_bn'],
          'paddle': ['PP-OCRv5_mobile_det|||PP-OCRv5_mobile_rec']}


class OCR:
    def __init__(self, config: Settings = settings):
        self.config: Settings = config
        if self.config.OCR_TYPE == 'doctr':
            det_name, reco_name = self.config.OCR_NAME.split('|||')
            self.model = DOCTR(det_name=det_name, reco_name=reco_name, device=self.config.OCR_DEVICE)
        elif self.config.OCR_TYPE == 'paddle':
            det_name, reco_name = self.config.OCR_NAME.split('|||')
            self.model = PaddleOCRModel(det_name=det_name, reco_name=reco_name, device=self.config.OCR_DEVICE)
        else:
            raise ValueError(f"Unsupported OCR type: {self.config.OCR_TYPE}")
        
    def save(self, results, output_path) -> Response:
        result = self.model.save(results, output_path)
        if isinstance(result, str):  # если результат - строка, значит произошла ошибка
            return Response(False, result, None)
        return Response(True, None, None)

    def process(self, images: List[Union[str, Image.Image, np.ndarray]]) -> Response:
        # Функция для распознавания текста на изображении. В данном случае она просто возвращает входные данные без изменений.
        # Имеется два варианта использования:
        # 1. Если передано только изображение, то будет распознан текст на нем.
        # 2. Если будет передан текст, то модель возьмет его как дополнительный контекст для распознавания.
        # Ограничения: если модель не использует текст, то второй вариант работать не будет.

        """
        result = [
        {
        'text': 'some text from image',
        'bbox': [x1, y1, x2, y2], # нормализованные координаты ограничивающего прямоугольника текста на изображении
        },
        {
        'text': 'some text from image',
        'bbox': [x1, y1, x2, y2], 
        }]
        """

        result = self.model.process(images)

        if isinstance(result, str):  # если результат - строка, значит произошла ошибка
            return Response(False, result, None)
        
        return Response(True, None, result)

