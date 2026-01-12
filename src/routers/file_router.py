import os
import asyncio
import logging
from starlette.status   import HTTP_500_INTERNAL_SERVER_ERROR
from fastapi            import APIRouter, HTTPException, UploadFile, Request, Depends, Form
from fastapi.responses  import FileResponse, JSONResponse, StreamingResponse

from src.config.app_config import settings
from src.utils.files_utils import get_file_extension_by_content_type
from src.utils.sse_utils import get_sse_headers
from src.utils.upload_utils import FileUploadService
from src.utils.sse_service_registry import sse_registry
from src.utils.sse_formatter import SSEEventFormatter

router = APIRouter(prefix='/files', tags=["files"])
logger = logging.getLogger(__name__)


@router.post("/upload/stream")
async def upload_file_stream(
    file: UploadFile,
    request: Request
) -> StreamingResponse:
    """
    Универсальная загрузка файла с SSE streaming прогресса.
    POST запрос сразу возвращает SSE поток.
    
    Архитектура:
    1. Валидация сессии
    2. Сохранение файла в temp директорию
    3. Прямой вызов ML сервиса через execute_stream() - БЕЗ RabbitMQ
    4. Проксирование SSE событий от сервиса к клиенту
    5. Сохранение метаданных в сессию
    
    Отправляет события:
    - event: progress - промежуточный прогресс от сервиса
    - event: complete - успешное завершение
    - event: error - ошибка обработки
    """
    logger.info(f"SSE upload started: {file.filename}")
    
    async def event_generator():
        session = None
        completed_successfully = False
        temp_file_path = None
        file_id = None
        
        file_service = FileUploadService()
        formatter = SSEEventFormatter()
        
        try:
            # Получаем сессию
            session = request.state.session.get_session()
            
            # 1. Валидация состояния сессии
            validation_error = file_service.validate_session_state(session)
            if validation_error:
                error_code, error_message = validation_error.split(":", 1)
                error_msg = {
                    "progress": -1,
                    "stage": "error",
                    "status": "error",
                    "error": {
                        "code": error_code,
                        "message": error_message,
                        "stage_failed": "validation",
                        "recoverable": True
                    }
                }
                yield formatter.format_event(error_msg)
                return
            
            # Устанавливаем pending только после успешной валидации
            session['pending'] = True
            
            # 2. Сохранение файла
            file_id, temp_file_path = await file_service.save_uploaded_file(file, session)
            
            # 3. Очистка предыдущих файлов
            await file_service.cleanup_previous_file(session)
            
            # 4. ПРЯМОЙ вызов ML сервиса через SSE registry (БЕЗ RabbitMQ!)
            file_name_without_ext = os.path.splitext(file.filename)[0] if file.filename else file_id
            
            logger.info(f"Starting ML service stream for: {file.filename}")
            
            # Проксируем все SSE события от ML сервиса напрямую
            async for sse_event in sse_registry.execute_service_stream(
                service_name="ml",
                params={
                    "data": {
                        "name": file_name_without_ext,
                        "path": temp_file_path
                    }
                }
            ):
                # Проверяем успешное завершение
                if 'event: complete' in sse_event:
                    completed_successfully = True
                
                yield sse_event
                await asyncio.sleep(0)
            
            # 5. Сохранение метаданных
            if completed_successfully:
                file_service.save_file_metadata(
                    session=session,
                    file_id=file_id,
                    filename=file.filename or "unknown",
                    file_path=temp_file_path,
                    content_type=file.content_type or "application/octet-stream",
                    size=file.size or 0
                )
                
                session['pending'] = False
                session['need_download'] = True
                
                logger.info(f"SSE upload completed successfully: {file.filename}")
                
        except Exception as e:
            logger.error(f"Critical error in SSE upload: {e}", exc_info=True)
            
            # Cleanup при критической ошибке
            if session:
                session['pending'] = False
                session['need_download'] = False
                if temp_file_path:
                    await file_service.cleanup_temp_file(temp_file_path, session)
            
            error_msg = {
                "progress": -1,
                "stage": "error",
                "status": "error",
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Критическая ошибка сервера",
                    "stage_failed": "streaming",
                    "error_details": str(e)
                }
            }
            yield formatter.format_event(error_msg)
            
        finally:
            # КРИТИЧЕСКИ ВАЖНО: Cleanup при разрыве SSE соединения
            if session and not completed_successfully:
                if session.get('pending', False):
                    logger.warning(f"SSE connection interrupted for file {file.filename}")
                    
                    session['pending'] = False
                    
                    # Удаляем файл только если обработка не завершена
                    if not session.get('need_download', False) and temp_file_path:
                        await file_service.cleanup_temp_file(temp_file_path, session)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=get_sse_headers()
    )