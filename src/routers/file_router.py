"""
Роутер для обработки файлов через S3 и ML сервисы.

Логика работы:
1. Получаем download_url (ссылка на файл в S3)
2. Скачиваем файл из S3 локально
3. Обрабатываем через ML сервис
4. Загружаем результат обратно на S3
5. Возвращаем ссылку на скачивание
"""

import os
import json
import asyncio
import logging
import shutil
from urllib.parse import urlparse
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config.services.ml_config import settings as ml_settings
from src.utils.sse_utils import get_sse_headers
from src.utils.sse_service_registry import sse_registry
from src.utils.sse_formatter import SSEEventFormatter

router = APIRouter(prefix='/files', tags=["files"])
logger = logging.getLogger(__name__)


class FileProcessRequest(BaseModel):
    """Схема запроса на обработку файла."""
    download_url: str = Field(
        ..., 
        description="URL файла в S3 хранилище для скачивания и обработки"
    )


def extract_object_key_from_url(url: str) -> Optional[str]:
    """
    Извлекает object_key из S3 URL.
    
    Поддерживаемые форматы:
    - https://storage.yandexcloud.net/bucket/path/to/file.mp4
    - https://bucket.storage.yandexcloud.net/path/to/file.mp4
    
    :param url: URL объекта в S3
    :return: object_key или None если не удалось извлечь
    """
    try:
        parsed = urlparse(url)
        path = parsed.path
        
        # Убираем ведущий слеш
        if path.startswith('/'):
            path = path[1:]
        
        # Если путь содержит имя бакета как первый сегмент, убираем его
        # Формат: /bucket/object_key или просто /object_key
        parts = path.split('/', 1)
        
        if len(parts) > 1:
            # Проверяем, есть ли бакет в хосте (virtual-hosted style)
            if '.storage.' in parsed.netloc:
                # bucket.storage.yandexcloud.net/object_key
                return path
            else:
                # storage.yandexcloud.net/bucket/object_key
                return parts[1]
        
        return path if path else None
        
    except Exception as e:
        logger.error(f"Failed to extract object_key from URL {url}: {e}")
        return None


def parse_sse_data(sse_event: str) -> Optional[dict]:
    """
    Парсит SSE событие и извлекает JSON данные.
    
    :param sse_event: SSE событие в виде строки
    :return: Распарсенные данные или None
    """
    try:
        for line in sse_event.split('\n'):
            if line.startswith('data: '):
                json_str = line[6:]  # Убираем "data: "
                return json.loads(json_str)
    except Exception as e:
        logger.debug(f"Failed to parse SSE data: {e}")
    return None


def is_complete_event(sse_event: str) -> bool:
    """Проверяет, является ли событие complete."""
    return 'event: complete' in sse_event


def is_error_event(sse_event: str) -> bool:
    """Проверяет, является ли событие ошибкой."""
    return 'event: error' in sse_event


@router.post("/stream")
async def process_file_stream(
    request_body: FileProcessRequest
) -> StreamingResponse:
    """
    Обработка файла из S3 с SSE streaming прогресса.
    
    Архитектура:
    1. Скачивание файла из S3 по download_url
    2. Обработка через ML сервис
    3. Загрузка результата обратно на S3
    4. Возврат ссылки на скачивание
    
    Отправляет события:
    - event: progress - промежуточный прогресс от сервисов
    - event: complete - успешное завершение с download_url
    - event: error - ошибка обработки
    """
    download_url = request_body.download_url
    logger.info(f"SSE processing started for URL: {download_url}")
    
    async def event_generator():
        formatter = SSEEventFormatter()
        downloaded_file_path: Optional[str] = None
        output_file_path: Optional[str] = None
        
        try:
            # === ЭТАП 1: Извлечение object_key из URL ===
            object_key = extract_object_key_from_url(download_url)
            
            if not object_key:
                error_msg = {
                    "progress": -1,
                    "stage": "error",
                    "status": "error",
                    "error": {
                        "code": "INVALID_URL",
                        "message": "Не удалось извлечь object_key из URL",
                        "stage_failed": "url_parsing",
                        "recoverable": True
                    }
                }
                yield formatter.format_event(error_msg)
                return
            
            logger.info(f"Extracted object_key: {object_key}")
            
            # === ЭТАП 2: Скачивание файла из S3 ===
            logger.info(f"Starting S3 download for: {object_key}")
            
            s3_download_result = None
            
            async for sse_event in sse_registry.execute_service_stream(
                service_name="ya_s3",
                params={
                    "data": {
                        "operation": "download",
                        "object_key": object_key
                    }
                }
            ):
                # Проверяем на ошибку
                if is_error_event(sse_event):
                    yield sse_event
                    return
                
                # Отлавливаем complete для получения download_path
                if is_complete_event(sse_event):
                    data = parse_sse_data(sse_event)
                    if data and 'result' in data:
                        s3_download_result = data['result']
                        downloaded_file_path = s3_download_result.get('download_path')
                        logger.info(f"S3 download complete: {downloaded_file_path}")
                    # Не отправляем complete событие от download, продолжаем обработку
                else:
                    # Проксируем progress события
                    yield sse_event
                
                await asyncio.sleep(0)
            
            if not downloaded_file_path:
                error_msg = {
                    "progress": -1,
                    "stage": "error",
                    "status": "error",
                    "error": {
                        "code": "DOWNLOAD_FAILED",
                        "message": "Не удалось скачать файл из S3",
                        "stage_failed": "s3_download",
                        "recoverable": True
                    }
                }
                yield formatter.format_event(error_msg)
                return
            
            # === ЭТАП 3: Обработка ML сервисом ===
            file_name = os.path.basename(downloaded_file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            logger.info(f"Starting ML processing for: {file_name}")
            
            ml_result = None
            
            async for sse_event in sse_registry.execute_service_stream(
                service_name="ml",
                params={
                    "data": {
                        "name": file_name_without_ext,
                        "path": downloaded_file_path
                    }
                }
            ):
                # Проверяем на ошибку
                if is_error_event(sse_event):
                    yield sse_event
                    return
                
                # Отлавливаем complete для получения output
                if is_complete_event(sse_event):
                    data = parse_sse_data(sse_event)
                    if data and 'result' in data:
                        ml_result = data['result']
                        output_filename = ml_result.get('output')
                        output_file_path = os.path.join(ml_settings.TEMP_DIR, output_filename)
                        logger.info(f"ML processing complete: {output_file_path}")
                    # Не отправляем complete событие от ML, продолжаем обработку
                else:
                    # Проксируем progress события
                    yield sse_event
                
                await asyncio.sleep(0)
            
            if not output_file_path or not os.path.exists(output_file_path):
                error_msg = {
                    "progress": -1,
                    "stage": "error",
                    "status": "error",
                    "error": {
                        "code": "ML_PROCESSING_FAILED",
                        "message": "ML сервис не вернул результат обработки",
                        "stage_failed": "ml_processing",
                        "recoverable": True
                    }
                }
                yield formatter.format_event(error_msg)
                return
            
            # === ЭТАП 4: Загрузка результата на S3 ===
            output_object_key = os.path.basename(output_file_path)
            
            logger.info(f"Starting S3 upload for: {output_file_path}")
            
            s3_upload_result = None
            
            async for sse_event in sse_registry.execute_service_stream(
                service_name="ya_s3",
                params={
                    "data": {
                        "operation": "upload",
                        "file_path": output_file_path,
                        "object_key": output_object_key
                    }
                }
            ):
                # Проверяем на ошибку
                if is_error_event(sse_event):
                    yield sse_event
                    return
                
                # Отлавливаем complete для получения public_url
                if is_complete_event(sse_event):
                    data = parse_sse_data(sse_event)
                    if data and 'result' in data:
                        s3_upload_result = data['result']
                        logger.info(f"S3 upload complete: {s3_upload_result.get('public_url')}")
                    # Не отправляем complete событие от upload, формируем финальный ответ
                else:
                    # Проксируем progress события
                    yield sse_event
                
                await asyncio.sleep(0)
            
            if not s3_upload_result:
                error_msg = {
                    "progress": -1,
                    "stage": "error",
                    "status": "error",
                    "error": {
                        "code": "UPLOAD_FAILED",
                        "message": "Не удалось загрузить результат на S3",
                        "stage_failed": "s3_upload",
                        "recoverable": True
                    }
                }
                yield formatter.format_event(error_msg)
                return
            
            # === ЭТАП 5: Формируем финальное событие с download_url ===
            # Используем presigned URL (download_url) вместо public_url для приватных бакетов
            result_download_url = s3_upload_result.get('download_url') or s3_upload_result.get('public_url')
            
            final_result = {
                "progress": 100,
                "stage": "complete",
                "status": "success",
                "result": {
                    "download_url": result_download_url,
                    "object_key": s3_upload_result.get('object_key'),
                    "size": s3_upload_result.get('size'),
                    "original_file": object_key,
                    "url_expires_in_hours": s3_upload_result.get('url_expires_in_hours')
                }
            }
            
            yield formatter.format_event(final_result)
            logger.info(f"Processing completed successfully. Download URL: {s3_upload_result.get('public_url')}")
            
        except Exception as e:
            logger.error(f"Critical error in SSE processing: {e}", exc_info=True)
            
            error_msg = {
                "progress": -1,
                "stage": "error",
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Критическая ошибка сервера",
                    "stage_failed": "streaming",
                    "error_details": str(e),
                    "recoverable": False
                }
            }
            yield formatter.format_event(error_msg)
            
        finally:
            # === ЭТАП 6: Очистка локальных файлов ===
            await cleanup_local_files(downloaded_file_path, output_file_path)


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=get_sse_headers()
    )


async def cleanup_local_files(*file_paths: Optional[str]):
    """
    Асинхронная очистка локальных файлов после обработки.
    
    :param file_paths: Пути к файлам для удаления
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                # Проверяем, это файл или директория
                if os.path.isfile(file_path):
                    await asyncio.to_thread(os.remove, file_path)
                    logger.info(f"Cleaned up file: {file_path}")
                elif os.path.isdir(file_path):
                    await asyncio.to_thread(shutil.rmtree, file_path)
                    logger.info(f"Cleaned up directory: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")
