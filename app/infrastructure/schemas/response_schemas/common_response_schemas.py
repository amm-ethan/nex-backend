import uuid
from typing import Any

from pydantic import BaseModel


class GeneralResponse(BaseModel):
    reference_no: str = str(uuid.uuid1())
    data: Any | None = None
    details: Any | None = None

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(*args, exclude_none=True, **kwargs)


class PagedResponse(BaseModel):
    total_pages: int
    current_page: int | None = 1
    total_count: int
    results: Any

    def dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(*args, exclude_none=True, **kwargs)


class HealthCheckResponse(BaseModel):
    is_alive: bool
    version: str
    root_path: str | None = None
