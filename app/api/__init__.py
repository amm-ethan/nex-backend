from fastapi import APIRouter

from app.api.endpoints.analytics_endpoint import router as analytics_endpoint
from app.core.config import settings

api_v1_router = APIRouter(prefix=settings.API_V1_STR)

api_v1_router.include_router(analytics_endpoint, tags=["analytics"])
