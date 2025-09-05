"""
Request schemas for infection detection API endpoints.
"""

from pydantic import BaseModel, Field


class InfectionDetectionRequest(BaseModel):
    """Request schema for infection detection analysis."""

    window_days: int = Field(
        14, ge=1, le=365, description="Contact window in days (1-365)"
    )
    date_origin: str | None = Field(
        None, description="Origin date for analysis (YYYY-MM-DD)"
    )
