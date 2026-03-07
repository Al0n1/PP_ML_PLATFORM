from typing import List, Union, Dict
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

from src.config.services.ml_config import settings, Settings
from src.services.ml_service.models.ocrs.base import BaseOCR


def ocr_to_dict(result) -> List[Dict[str, Union[str, List[float]]]]:
    data = []
    for res in result:
        boxes = res['rec_boxes']
        texts = res['rec_texts']
        h, w, _ = res['doc_preprocessor_res']['output_img'].shape
        sample = []
        for text, box in zip(texts, boxes):
            box = np.array(box)
            if box.ndim == 1:
                # Flat format: [x1, y1, x2, y2, ...] → reshape to Nx2
                box = box.reshape(-1, 2)
            x_min = float(np.min(box[:, 0])) / w
            y_min = float(np.min(box[:, 1])) / h
            x_max = float(np.max(box[:, 0])) / w
            y_max = float(np.max(box[:, 1])) / h
            sample.append({
                "text": text,
                "bbox": [x_min, y_min, x_max, y_max]
            })
        data.append(sample)
    return data    


class PaddleOCRModel(BaseOCR):
    def __init__(self, device: str = None, det_name: str = "PP-OCRv5_mobile_det", reco_name: str = "PP-OCRv5_mobile_rec"):
        super().__init__(device=device, name='paddle')

        self.device = device.replace('cuda', 'gpu')
        self.model = PaddleOCR(
            text_detection_model_name=det_name,
            text_recognition_model_name=reco_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,)

    def process(self, images: List[Union[str, Image.Image, np.ndarray]]):
        try:
            result = self.model.predict(images)
            return ocr_to_dict(result)
        except Exception as e:
            return str(e)
