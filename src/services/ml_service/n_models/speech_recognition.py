import whisper
from ..utils import Response
class SimpleWhisper:
    """
    Простой класс для инициализации модели Whisper и распознавания аудио.
    Использует локальную библиотеку `whisper`.
    """

    def __init__(self, model_name: str = "small", device: str = "cpu"):
        if model_name not in whisper.available_models():
            raise ValueError(f"Model '{model_name}' is not available. Choose from: {whisper.available_models()}")
        self.model = whisper.load_model(model_name, device=device)

    def transcribe(self, audio_path: str, task: str = "translate") -> str:
        """Возвращает только распознанный текст.
        
        Args:
            audio_path: путь к аудиофайлу.
            task: "transcribe" — вывод на языке оригинала,
                  "translate"  — всегда выводит английский текст
                  (по умолчанию "translate", т.к. далее по пайплайну EN→RU перевод).
        """
        try:
            result = self.model.transcribe(audio_path, task=task)
            text = result.get("text", "")
            return Response(True, None, text)
        except Exception as e:
            return Response(False, e, None)
    