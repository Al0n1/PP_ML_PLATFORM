import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config.app_config import settings
from src.config.logging_config import setup_logging
from src.routes import get_apps_router
from src.services.file_pipeline_orchestrator import FilePipelineOrchestrator
from src.services.ml_service.ml_service import MLService
from src.services.ya_s3_service import YaS3Service

# Инициализация системы логирования
setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Initializing application dependencies")

    ml_service = MLService()
    s3_service = YaS3Service()

    application.state.ml_service = ml_service
    application.state.s3_service = s3_service
    application.state.file_pipeline = FilePipelineOrchestrator(
        ml_service=ml_service,
        s3_service=s3_service,
    )

    logger.info("Application dependencies initialized")
    yield
    logger.info("Application shutdown complete")


def get_application() -> FastAPI:
    logger.info("Initializing FastAPI application")
    application = FastAPI(
        title=settings.PROJECT_NAME,
        debug=settings.DEBUG,
        version=settings.VERSION,
        lifespan=lifespan,
    )
    
    application.include_router(get_apps_router())

    logger.info("FastAPI application initialized successfully")
    
    return application


app = get_application()


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting application server")
    uvicorn.run("src.app:app", host="0.0.0.0", port=settings.APP_PORT)
