from fastapi import APIRouter

from app.api.endpoints.infection_detection_endpoint import (
    router as infection_detection_endpoint,
)
from app.core.config import settings

api_v1_router = APIRouter(prefix=settings.API_V1_STR)

api_v1_router.include_router(
    infection_detection_endpoint,
    prefix="/infection-detection",
    tags=["infection-detection"],
)
