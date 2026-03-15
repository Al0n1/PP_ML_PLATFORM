from ...utils import VideoData

class BaseSpeechModel:
    def __init__(self, **kwargs):
        pass

    def process(self, video_data: VideoData) -> VideoData:
        """Обрабатывает видео данные и возвращает обновлённые данные.

        Args:
            video_data: Входные данные видео, включая пути к файлам и распознанный текст.

        Returns:
            Обновлённые данные видео с добавленными результатами обработки.
        """
        raise NotImplementedError("Метод process должен быть реализован в подклассе.")