class BaseTranslationModel:
    def __init__(self, image_support: bool = False):
        self.image_support = image_support
    
    def __call__(self, text: str):
        return {'status': True, 'error': 'Error description', 'text': 'some text', 'source_text': text}