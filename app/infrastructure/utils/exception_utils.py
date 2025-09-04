import sys
import traceback
from logging import Logger
from typing import Any

from fastapi import HTTPException, status


class CustomHttpError(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        headers: dict[str, Any] | None = None,
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)


class CustomLockedHttpError(CustomHttpError):
    def __init__(self, detail: list[dict[str, Any]]):
        super().__init__(status_code=status.HTTP_423_LOCKED, detail=detail)


class CustomBadRequestHttpError(CustomHttpError):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class CustomGatewayTimeoutError(CustomHttpError):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail)


class CustomNotFoundHttpError(CustomHttpError):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class CustomUnauthorizedHttpError(CustomHttpError):
    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class CustomUnauthorizedWithHeaderHttpError(CustomHttpError):
    def __init__(self, detail: str | None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class CustomUnprocessableEntityHttpError(CustomHttpError):
    def __init__(self, detail: list[dict[str, Any]] | dict[str, Any]):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail
        )


def log_exception_details(logger: Logger) -> None:
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb = traceback.extract_tb(exc_tb)
    if tb:
        filename, lineno, func, text = tb[-1]
        logger.error(f"Exception in {filename}, line {lineno}, in {func}")
        logger.error(f"Code: {text}")
