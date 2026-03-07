import re
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wavfile
from ..utils import Response

class TextToSpeech:
    def __init__(self, model_name="facebook/mms-tts-rus", device=None):
        """
        Инициализация модели TTS и токенизатора.
        
        Args:
            model_name (str): название предобученной модели.
            device (str, optional): устройство для работы модели ('cpu' или 'cuda'). 
                                     Если None, автоматически выбирается CUDA если доступна.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VitsModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sampling_rate = self.model.config.sampling_rate

    def synthesize(self, text, output_path="output.wav"):
        """
        Синтезирует аудио из текста и сохраняет его в файл.
        
        Args:
            text (str): текст для синтеза.
            output_path (str): путь для сохранения WAV-файла.
        """
        try:
            if not text or not text.strip():
                return Response(False, "TTS input text is empty", None)

            # Check that the text contains at least some word characters
            # (not just punctuation/dots which the tokenizer may discard entirely)
            if not re.search(r'\w', text):
                return Response(
                    False,
                    f"TTS input text contains no usable characters: {text[:100]!r}",
                    None,
                )

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            # Guard against empty input_ids after tokenization
            if inputs.input_ids.numel() == 0 or inputs.input_ids.shape[-1] == 0:
                return Response(
                    False,
                    f"TTS tokenizer produced empty input_ids for text: {text[:100]!r}",
                    None,
                )

            with torch.no_grad():
                waveform = self.model(**inputs).waveform

            # Преобразуем тензор в numpy и сохраняем в WAV
            waveform_np = waveform.squeeze().cpu().numpy()
            wavfile.write(output_path, rate=self.sampling_rate, data=waveform_np)
            return Response(True, None, None)
        except Exception as e:
            return Response(False, e, None)

# Пример использования
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.synthesize("Пример текста на русском языке", output_path="techno.wav")
