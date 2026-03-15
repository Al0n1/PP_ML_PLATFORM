from typing import List, Union, Dict
from PIL import Image
import numpy as np
import logging
from paddleocr import PaddleOCR


from ...utils import VideoData, Frame, TextItem, BoundingBox
from src.services.ml_service.models.ocrs.base import BaseOCR

logger = logging.getLogger(__name__)


def ocr_to_dict(result, video_data: VideoData) -> VideoData:
    frames = []
    for res in result:
        boxes = res['rec_boxes']
        texts = res['rec_texts']
        h, w, _ = res['doc_preprocessor_res']['output_img'].shape
        text_items = []

        box_items = []
        texts_items = []
        for text, box in zip(texts, boxes):
            box = np.array(box)
            if box.ndim == 1:
                box = box.reshape(-1, 2)
            x_min = float(np.min(box[:, 0])) / w
            y_min = float(np.min(box[:, 1])) / h
            x_max = float(np.max(box[:, 0])) / w
            y_max = float(np.max(box[:, 1])) / h
            texts_items.append(text)
            text_items.append(TextItem(
                text=text,
                bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            ))
        frames.append(Frame(texts=text_items,
                            bounding_boxes=box_items,
                            source_texts=texts_items))
    video_data.video.ocr_frames = frames
    return video_data    


class PaddleOCRModel(BaseOCR):
    def __init__(self,
                 device: str = None,
                 det_name: str = "PP-OCRv5_mobile_det",
                 reco_name: str = "PP-OCRv5_mobile_rec"):
        super().__init__(device=device, name='paddle')

        self.device = device.replace('cuda', 'gpu')
        self.model = PaddleOCR(
            text_detection_model_name=det_name,
            text_recognition_model_name=reco_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=self.device,)

    def process(self, video_data: VideoData) -> VideoData:
        try:
            images = [video_data.video.source_frames[idx] for idx in video_data.video.selected_frames]
            result = self.model.predict(images)
            return ocr_to_dict(result, video_data)
        except Exception as e:
            logger.error(f"OCR:{self.name} processing error: {e}")
            return video_data
