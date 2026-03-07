from typing import List, Dict, Union
from PIL import Image
import numpy as np
import json


class BaseOCR:
    def __init__(self, device: str = None):
        self.device = device

    def process(self, images: List[Union[str, Image.Image, np.ndarray]]) -> List[Dict[str, Union[str, List[float]]]]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, results: List[Dict[str, Union[str, List[float]]]], output_path: str) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, results: List[Dict[str, Union[str, List[float]]]], output_path: str) -> bool|str:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            return str(e)
    