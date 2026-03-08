from typing import List, Any, Union

class BaseRecognizer:
    def __init__(self, name: str = 'base_recognizer'):
        self.name = name

    def process(self, data: Union[str, Any]) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    