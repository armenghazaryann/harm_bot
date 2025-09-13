import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic_core import _pydantic_core
from sqlalchemy import text
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

from api.di.container import ApplicationContainer as DependencyContainer
from core.settings import SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("rag")


class CustomFastAPI(FastAPI):
    container: DependencyContainer


@asynccontextmanager
async def lifespan(_app: CustomFastAPI):
    logger.info("Starting application initialization...")
    start_time = time.time()

    try:
        logger.info("Initializing database connection...")
        db_start = time.time()
        # Get the database resource and initialize it
        db_resource = _app.container.infrastructure.database()
        await db_resource.init()
        async with db_resource.engine.begin() as _conn:
            await _conn.execute(text("SET lock_timeout = '4s'"))
            await _conn.execute(text("SET statement_timeout = '8s'"))
            # Verify database connection
            await _conn.execute(text("SELECT 1"))
            logger.info(
                f"✅ Database connection established in {time.time() - db_start:.2f}s"
            )

        logger.info("Initializing Redis connection...")
        redis_start = time.time()
        redis_resource = _app.container.infrastructure.redis_db()
        await redis_resource.init()
        await redis_resource.connect()
        logger.info(
            f"✅ Redis connection established in {time.time() - redis_start:.2f}s"
        )

        logger.info("Initializing MinIO client...")
        minio_start = time.time()
        minio_resource = _app.container.infrastructure.minio_client()
        await minio_resource.init()
        await minio_resource.ensure_bucket()
        logger.info(f"✅ MinIO client initialized in {time.time() - minio_start:.2f}s")

        logger.info(
            f"✅ Application startup completed in {time.time() - start_time:.2f}s"
        )
    except Exception as e:
        logger.exception(f"❌ Failed to initialize application: {str(e)}")
        raise

    yield

    try:
        # Cleanup connections
        redis_resource = _app.container.infrastructure.redis_db()
        if redis_resource:
            await redis_resource.disconnect()
        db_resource = _app.container.infrastructure.database()
        if db_resource:
            await db_resource.shutdown()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.exception("Error during shutdown", str(e))


def create_fastapi_app() -> CustomFastAPI:
    origins = {
        "*",
        "http://localhost",
        "http://localhost:*",
        "http://localhost:3000",
        "http://localhost:8000",
    }

    _app = CustomFastAPI(
        title="RAG ETL API",
        description="Production-grade RAG ETL Pipeline for Financial Documents",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize dependency container
    _app.container = DependencyContainer()
    _app.container.infrastructure.config.from_dict(SETTINGS.model_dump())
    _app.container.wire(modules=[sys.modules[__name__]])
    _app.container.init_resources()

    # Disable default uvicorn logging
    logging.getLogger("uvicorn.error").disabled = False
    logging.getLogger("uvicorn.access").disabled = False

    # Add CORS middleware
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include feature routers
    from api.features.documents.router import router as documents_router
    from api.features.query.router import router as query_router

    _app.include_router(
        documents_router, prefix="/api/v1/documents", tags=["Documents"]
    )
    _app.include_router(query_router, prefix="/api/v1/query", tags=["Query"])

    return _app


app = create_fastapi_app()


# Health check endpoints
@app.get("/")
async def root():
    return {"message": "RAG ETL API is running", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    return {"status": "ok"}


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"{exc.detail} : {request.url}",
            "status_code": 404,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": str(exc), "status_code": 422},
    )


@app.exception_handler(_pydantic_core.ValidationError)
async def pydantic_validation_handler(
    request: Request, exc: _pydantic_core.ValidationError
):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "detail": str(exc), "status_code": 422},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception", str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "status_code": 500,
        },
    )
