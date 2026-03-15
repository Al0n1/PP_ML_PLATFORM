import cv2
import numpy as np
import os
from .utils import Response, VideoData
# try:
#     from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip, VideoClip
# except ImportError:
#     print("moviepy is not installed. Audio extraction will not work.")
try:
    from moviepy import ImageSequenceClip, AudioFileClip, VideoFileClip
except ImportError:
    print("moviepy is not installed. Audio extraction will not work.")


def extract_frames(video_data: VideoData) -> VideoData:
    cap = cv2.VideoCapture(video_data.source_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # Попытка открыть снова без указания backend
        cap = cv2.VideoCapture(video_data.source_path)
        if not cap.isOpened():
            return video_data
    # Успешное открытие видео
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    video_data.video.source_frames = frames
    return video_data


def extract_audio(video_path, output_audio_path):
    """
    Извлекает аудио из видео и сохраняет в файл.

    :param video_path: путь к видеофайлу
    :param output_audio_path: путь для сохранения аудиофайла
    """
    try:
        with VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(output_audio_path)
            # TODO: удалить временный аудиофайл после использования, если он не нужен
            # audio.write_audiofile(f'var/data/audio/audio.mp3')
        return Response(True, None, None)
    except Exception as e:
        return Response(False, str(e), None)
        

def create_video_with_new_audio(video_data: VideoData) -> Response:
    """
    Создает новое видео из изображений и нового аудио.
    """
    try:
        output_video_path = os.path.join(video_data.temp_dir, f'{video_data.video_name}.mp4')
        original_video_path = video_data.source_path
        new_audio_path = video_data.audio.output_audio_path
        count_frames = len(video_data.video.source_frames)
        with VideoFileClip(original_video_path) as orig_video, AudioFileClip(new_audio_path) as audio_clip:
            video_size = orig_video.size
            audio_duration = audio_clip.duration
            frame_duration = audio_duration / count_frames
            # video_clip = ImageSequenceClip(image_files, durations=[frame_duration]*len(image_files))
            video_clip: ImageSequenceClip = ImageSequenceClip(video_data.video.source_frames,
                                           durations=[frame_duration]*count_frames)
            video_clip = video_clip.resized(new_size=video_size)
            video_clip = video_clip.with_audio(audio_clip)
            video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        return Response(True, None, None)
    except Exception as e:
        return Response(False, str(e), None)