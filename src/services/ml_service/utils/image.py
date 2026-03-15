import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL import ImageFont
import os
from tqdm import tqdm
import logging

from typing import List, Union
from .utils import Response, Frame, TextItem, BoundingBox, VideoData

logger = logging.getLogger(__name__)

def get_font(font_path=None, font_size=20):
    """
    Возвращает объект шрифта PIL с поддержкой русского языка.
    Если font_path не указан, использует стандартные пути для MacOS, Linux, Windows.
    """
    possible_paths = []

    if font_path:
        possible_paths.append(font_path)
    
    # Пути по умолчанию для MacOS
    possible_paths += [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/DejaVuSans.ttf"
    ]
    # Пути по умолчанию для Linux
    possible_paths += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]
    # Пути по умолчанию для Windows
    possible_paths += [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)

    raise OSError("Не найден шрифт для русского текста. Укажите font_path с .ttf файлом")

def draw_translations_on_image(image: Image.Image, frame: Frame, font_path=None) -> Image.Image:
    """
    Закрашивает текст на изображении и добавляет русский перевод.

    Args:
        image (PIL.Image.Image): объект изображения PIL
        frame (Frame): кадр с распознанными текстами и переводами
        font_path (str, optional): путь к .ttf шрифту с поддержкой кириллицы

    Returns:
        PIL.Image.Image: изображение с наложенным текстом
    """
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    if not frame.texts:
        return img

    for item in frame.texts:
        if not item.bounding_box:
            continue

        text = item.translation or item.text or ""
        if not text:
            continue

        bb = item.bounding_box
        x1 = int(bb.x_min * img_width)
        y1 = int(bb.y_min * img_height)
        x2 = int(bb.x_max * img_width)
        y2 = int(bb.y_max * img_height)

        draw.rectangle([x1, y1, x2, y2], fill="white")

        # Подбор размера шрифта
        font_size = 10
        font = get_font(font_path=font_path, font_size=font_size)
        while True:
            bbox_text = draw.textbbox((0, 0), text, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            if text_width >= (x2 - x1) or text_height >= (y2 - y1):
                font_size -= 1
                font = get_font(font_path=font_path, font_size=font_size)
                break
            font_size += 1
            font = get_font(font_path=font_path, font_size=font_size)

        draw.text((x1, y1), text, fill="black", font=font)

    return img

def translate_images(video_data: VideoData, output_dir: str, font_path="arial.ttf") -> VideoData:
    """
    Обрабатывает кадры видео, накладывая переведённый текст поверх изображений.

    Args:
        video_data (VideoData): объект VideoData с изображениями, OCR-кадрами и маппингом индексов
        output_dir (str): папка для сохранения обработанных изображений
        font_path (str): путь к .ttf шрифту, поддерживающему русский язык
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        frames = video_data.video.source_frames
        indexes = video_data.video.translated_frames_indexes
        ocr_frames = video_data.video.ocr_frames

        for idx, frame_img in enumerate(tqdm(
            frames,
            desc="Translating images",
            unit="img",
            ncols=100,
            colour="blue",
        )):
            annotation = ocr_frames[indexes[idx]]

            if isinstance(frame_img, np.ndarray):
                image = Image.fromarray(frame_img)
            else:
                image = Image.open(frame_img)

            result_img = draw_translations_on_image(image, annotation, font_path=font_path)

            filename = f"frame_{idx:06d}.jpg"
            result_img.save(os.path.join(output_dir, filename))
    except Exception as e:
        logger.error(f"Error in translate_images: {e}")

    return video_data


