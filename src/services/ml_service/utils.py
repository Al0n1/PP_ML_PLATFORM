import cv2
import os
from PIL import Image
import logging 
from typing import List, Dict
from moviepy import VideoFileClip, AudioFileClip, ImageSequenceClip
import wave
import contextlib
from audiostretchy.stretch import stretch_audio
from pydub import AudioSegment
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def rename_file(directory, old_name, new_name):
    old_path = os.path.join(directory, old_name)
    new_path = os.path.join(directory, new_name)
    
    try:
        os.rename(old_path, new_path)
        return {'status': True}
    except FileNotFoundError:
        return {'status': False, 'error': f"Файл '{old_name}' не найден в директории '{directory}'."}
    except FileExistsError:
        return {'status': False, 'error': f"Файл с именем '{new_name}' уже существует в директории '{directory}'."}
    except Exception as e:
        return {'status': False, 'error': f"Произошла ошибка: {e}"}

def extract_audio_from_video(video_path, output_audio_path):
    """
    Извлекает аудио из видеофайла и сохраняет его.

    :param video_path: путь к исходному видеофайлу
    :param output_audio_path: путь для сохранения извлеченного аудио
    """
    print(f"extract_audio_from_video: {video_path}")
    try:
        with VideoFileClip(video_path) as video:
            audio = video.audio
            audio.write_audiofile(output_audio_path)   
        return {'status': True}
    except Exception as e:
        return {'status': False, 'error': e}
        

def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def video_to_frames(video_path: str, output_folder: str) -> None:
    safe_makedirs(output_folder)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:08d}.jpg"),
                    frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_count += 1
    cap.release()
    logging.info(f"Извлечено {frame_count} кадров → {output_folder}")
    

def extract_audio_from_video(video_path: str, output_audio_path: str) -> dict:
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        return {'status': False, 'error': ValueError("В видео нет аудиодорожки")}
    clip.audio.write_audiofile(output_audio_path)
    clip.close()
    logging.info(f"Аудио сохранено: {output_audio_path}")
    return {'status': True}


def list_images_sorted(dir_path: str) -> List[str]:
    """Сортировка изображений по имени файла"""
    return sorted(
        [os.path.join(dir_path, f) for f in os.listdir(dir_path)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=lambda p: os.path.basename(p)
    )


def images_to_video_with_audio_auto_fps(
    images_dir: str,
    audio_path: str,
    output_path: str,
    source_video_path: str,
    temp_dir: str = 'var/temp'
) -> dict:
    """
    Создаёт видео из кадров и синхронизирует аудио по длине исходного видео.
    Если длительность аудио и видео различается — аудио ускоряется или замедляется.
    """

    # Загружаем исходное видео для получения FPS и длительности
    source_clip = VideoFileClip(source_video_path)
    fps = source_clip.fps
    target_duration = source_clip.duration
    logging.info(f"FPS исходного видео: {fps}, длительность: {target_duration:.2f} сек")

    # Список изображений
    images = list_images_sorted(images_dir)
    if not images:
        raise ValueError("Не найдено изображений для сборки видео")

    # Создаём видеоклип из изображений
    clip = ImageSequenceClip(images, fps=fps)

    # Подгоняем длительность видеоряда под исходное видео
    if abs(clip.duration - target_duration) > 0.05:
        clip = clip.with_duration(target_duration)
        logging.info(f"Продолжительность кадрового видео подогнана под оригинал ({target_duration:.2f} сек)")

    # Обработка аудио
    if audio_path and os.path.exists(audio_path):
        file_name = os.path.basename(audio_path.replace('.mp3', '').replace('.wav', ''))
        wav_path = f"{temp_dir}/temp_{file_name}_audio.wav"
        mp3_path = f'{temp_dir}/temp_{file_name}_audio.mp3'
        # Пример использования:
        change_audio_speed(audio_path, wav_path, desired_duration_sec=target_duration)
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3")
        audio_clip = AudioFileClip(mp3_path)
        original_audio_duration = audio_clip.duration
        clip = clip.with_audio(audio_clip)

    # Экспорт финального видео
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
    logging.info(f"Видео сохранено: {output_path}")

    # Очистка ресурсов
    clip.close()
    source_clip.close()

    return {'status': True}



def change_audio_speed(input_path: str,
                       output_path: str,
                       desired_duration_sec: float,
                       temp_dir: str = 'var/temp') -> Dict[str, float]:
    """
    Изменяет длительность WAV-аудио без изменения высоты тона
    и обрезает результат до заданной длительности.
    
    Параметры:
        input_path (str): путь к исходному WAV-файлу.
        output_path (str): путь, куда сохранить изменённое аудио.
        desired_duration_sec (float): целевая длительность в секундах.
    """
    # --- Определяем исходную длительность ---
    with contextlib.closing(wave.open(input_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        original_duration = frames / float(rate)

    # --- Рассчитываем коэффициент изменения времени ---
    ratio = desired_duration_sec / original_duration
    file_name = os.path.basename(input_path.replace('.mp3', '').replace('.wav', ''))
    # --- Применяем time-stretch ---
    temp_output = f"{temp_dir}/temp_{file_name}_stretched.wav"
    stretch_audio(input_path, temp_output, ratio=ratio)

    # --- Обрезаем до нужной длины ---
    audio = AudioSegment.from_wav(temp_output)
    target_ms = int(desired_duration_sec * 1000)
    trimmed_audio = audio[:target_ms]  # аккуратно обрезаем

    # --- Сохраняем итоговый результат ---
    trimmed_audio.export(output_path, format="wav")

    # print(f"✅ Исходная длительность: {original_duration:.2f} сек")
    # print(f"🎯 Целевая длительность: {desired_duration_sec:.2f} сек")
    # print(f"💾 Файл сохранён: {output_path}")
    # print(f"⚙️ Коэффициент изменения времени (ratio): {ratio:.3f} "
    #       f"({'замедление' if ratio > 1 else 'ускорение'})")
    # print(f"✂️ Итоговый файл обрезан до {desired_duration_sec:.2f} секунд")

    return {'status': True,
            'original_duration_sec': original_duration,
            'target_duration_sec': desired_duration_sec,
            'ratio': ratio}


