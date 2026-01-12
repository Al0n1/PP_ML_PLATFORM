"""
Роутер для обработки файлов через S3 и ML сервисы.

Логика работы:
1. Получаем download_url (ссылка на файл в S3, может быть presigned URL)
2. Скачиваем файл локально (по HTTP если presigned URL, или через S3 API)
3. Обрабатываем через ML сервис
4. Загружаем результат обратно на S3
5. Возвращаем ссылку на скачивание
"""

import os
import json
import asyncio
import logging
import shutil
from urllib.parse import urlparse, parse_qs
from typing import Optional, Tuple
from pathlib import Path

import httpx
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


def is_presigned_url(url: str) -> bool:
    """
    Проверяет, является ли URL presigned URL (содержит параметры подписи AWS/S3).
    
    Presigned URL содержит параметры:
    - X-Amz-Signature - подпись запроса
    - X-Amz-Credential - учетные данные
    - X-Amz-Algorithm - алгоритм подписи
    
    :param url: URL для проверки
    :return: True если это presigned URL
    """
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Проверяем наличие характерных параметров AWS подписи
        signature_params = ['X-Amz-Signature', 'X-Amz-Credential', 'X-Amz-Algorithm']
        return any(param in query_params for param in signature_params)
    except Exception:
        return False


async def download_file_from_url(
    url: str, 
    dest_dir: str,
    formatter: SSEEventFormatter
) -> Tuple[Optional[str], int]:
    """
    Скачивает файл напрямую по HTTP URL (presigned URL).
    
    :param url: URL для скачивания (presigned URL)
    :param dest_dir: Директория для сохранения файла
    :param formatter: SSE форматтер для отправки сообщений о прогрессе
    :return: Tuple (путь к файлу, размер файла) или (None, 0) при ошибке
    :yields: SSE события с прогрессом
    """
    # Создаем директорию если не существует
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    
    # Извлекаем имя файла из URL
    filename = extract_object_key_from_url(url)
    if not filename:
        filename = "downloaded_file"
    
    file_path = os.path.join(dest_dir, filename)
    
    logger.info(f"Starting HTTP download: {url[:100]}... -> {file_path}")
    
    timeout = httpx.Timeout(
        connect=30.0,
        read=300.0,  # 5 минут на чтение
        write=30.0,
        pool=30.0
    )
    
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream('GET', url) as response:
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            logger.info(f"HTTP response: status={response.status_code}, content-length={total_size}")
            
            with open(file_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
            
            logger.info(f"HTTP download complete: {file_path} ({downloaded_size} bytes)")
            
            return file_path, downloaded_size


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
            
            # === ЭТАП 2: Скачивание файла ===
            # Проверяем, является ли URL presigned (с подписью) или обычный S3 URL
            use_http_download = is_presigned_url(download_url)
            
            if use_http_download:
                # === Скачивание по HTTP (presigned URL) ===
                logger.info(f"Detected presigned URL, using HTTP download for: {object_key}")
                
                # Отправляем SSE прогресс о начале скачивания
                progress_msg = {
                    "progress": 5,
                    "stage": "downloading",
                    "status": "in_progress",
                    "message": "Скачивание файла по presigned URL..."
                }
                yield formatter.format_event(progress_msg)
                
                try:
                    # Используем временную директорию из ml_settings
                    temp_dir = os.getenv("TEMP_DIR", "var/temp")
                    downloaded_file_path, file_size = await download_file_from_url(
                        url=download_url,
                        dest_dir=temp_dir,
                        formatter=formatter
                    )
                    
                    if not downloaded_file_path:
                        raise Exception("Не удалось скачать файл")
                    
                    logger.info(f"HTTP download complete: {downloaded_file_path} ({file_size} bytes)")
                    
                    # Отправляем SSE прогресс о завершении скачивания
                    progress_msg = {
                        "progress": 15,
                        "stage": "downloading",
                        "status": "completed",
                        "message": f"Файл скачан: {file_size} байт"
                    }
                    yield formatter.format_event(progress_msg)
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP download failed with status {e.response.status_code}: {e}")
                    error_msg = {
                        "progress": -1,
                        "stage": "error",
                        "status": "error",
                        "error": {
                            "code": "HTTP_DOWNLOAD_FAILED",
                            "message": f"Ошибка скачивания: HTTP {e.response.status_code}",
                            "stage_failed": "http_download",
                            "recoverable": True
                        }
                    }
                    yield formatter.format_event(error_msg)
                    return
                except Exception as e:
                    logger.error(f"HTTP download failed: {e}", exc_info=True)
                    error_msg = {
                        "progress": -1,
                        "stage": "error",
                        "status": "error",
                        "error": {
                            "code": "HTTP_DOWNLOAD_FAILED",
                            "message": f"Ошибка скачивания по HTTP: {str(e)}",
                            "stage_failed": "http_download",
                            "error_details": str(e),
                            "recoverable": True
                        }
                    }
                    yield formatter.format_event(error_msg)
                    return
            else:
                # === Скачивание через S3 API (обычный URL без подписи) ===
                logger.info(f"Using S3 API download for: {object_key}")
                
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
                        "message": "Не удалось скачать файл",
                        "stage_failed": "download",
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
                service_name="ml.execute",
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
                
                # Отлавливаем complete для получения download_url
                if is_complete_event(sse_event):
                    data = parse_sse_data(sse_event)
                    if data and 'result' in data:
                        s3_upload_result = data['result']
                        logger.info(f"S3 upload complete: {s3_upload_result.get('download_url')}")
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
            # Используем presigned URL (download_url) вместо download_url для приватных бакетов
            result_download_url = s3_upload_result.get('download_url') or s3_upload_result.get('download_url')
            
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
            logger.info(f"Processing completed successfully. Download URL: {s3_upload_result.get('download_url')}")
            
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
