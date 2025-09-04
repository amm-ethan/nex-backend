import logging.config
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_v1_router
from app.core.config import logger_config, settings
from app.infrastructure.schemas.response_schemas.common_response_schemas import (
    HealthCheckResponse,
)
from app.infrastructure.utils.exception_handler_utils import (
    add_http_exception_handler,
    add_server_exception_handler,
    add_validation_error_handler,
)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logging.config.dictConfig(logger_config)
    yield


# Create the FastAPI application instance with CORS middleware and exception handlers.
app = FastAPI(
    title=settings.PROJECT_NAME, root_path=settings.ROOT_PATH, lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1_router)  # Include API routers

# Add exception handlers for validation, HTTP, and server errors
add_validation_error_handler(app)
add_http_exception_handler(app)
add_server_exception_handler(app)


# Health check endpoint to verify the application is running.
@app.get("/", response_model=HealthCheckResponse)
async def health_check(request: Request) -> HealthCheckResponse:
    return HealthCheckResponse(
        is_alive=True,
        version=settings.VERSION,
        root_path=request.scope.get("root_path"),
    )
