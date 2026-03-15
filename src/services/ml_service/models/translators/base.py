from ...utils import VideoData, Audio, Video

class BaseTranslationModel:
    def __init__(self, image_support: bool = False):
        self.image_support = image_support

    def process(self, video_data: VideoData) -> VideoData:
        if video_data.audio.source_text:
            video_data = self._process_audio(video_data)
        if video_data.video.source_frames and video_data.video.ocr_frames:
            video_data = self._process_video(video_data)
        return video_data
    
    def _process_audio(self, video_data: VideoData) -> VideoData:
        # Базовая реализация для обработки аудио, может быть переопределена в наследниках
        raise NotImplementedError("Audio processing not implemented in BaseTranslationModel")
    
    def _process_video(self, video_data: VideoData) -> VideoData:
        # Базовая реализация для обработки видео, может быть переопределена в наследниках
        raise NotImplementedError("Video processing not implemented in BaseTranslationModel")