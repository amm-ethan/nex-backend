import logging
import traceback
from collections.abc import Awaitable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from app.infrastructure.schemas.response_schemas.common_response_schemas import (
    GeneralResponse,
)

logger = logging.getLogger(__name__)


def add_http_exception_handler(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        _: Request, exc: HTTPException
    ) -> Response | Awaitable[JSONResponse]:
        """
        Handles http related errors and returns a custom JSON response.

        Args:
            _ (Request): The incoming HTTP request that caused the validation error.
            exc (HTTPException): The http exception containing error details.

        Returns:
            Response | Awaitable[JSONResponse]: A JSONResponse with error details and HTTP 400 status code.
        """
        response_body = GeneralResponse(details=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=response_body.dict())


def add_server_exception_handler(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def server_exception_handler(
        _: Request, exc: Exception
    ) -> Response | Awaitable[JSONResponse]:
        """
        Handles http related errors and returns a custom JSON response.

        Args:
            _ (Request): The incoming HTTP request that caused the validation error.
            exc (Exception): The exception (unhandled) containing error details.

        Returns:
            Response | Awaitable[JSONResponse]: A JSONResponse with error details and HTTP 500 status code.
        """
        response_body = GeneralResponse(
            data=f"An internal server error occurred. {type(exc).__name__}",
            details=str(exc),
        )

        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_str = "".join(tb).strip()

        logger.error(
            f"Reference: {response_body.reference_no}\n {type(exc).__name__} \n {tb_str}"
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_body.dict(),
        )


def add_validation_error_handler(app: FastAPI) -> None:
    """
    Adds a custom handler for RequestValidationError to the FastAPI application.

    Args:
        app (FastAPI): The FastAPI application instance to which the error handler will be added.

    This function defines and registers an exception handler that processes validation errors raised
    during request validation in FastAPI. When a validation error occurs, this handler formats the
    error details, logs the error, and returns a standardized JSON response.
    """

    # Define the exception handler for RequestValidationError
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, exc: RequestValidationError
    ) -> Response | Awaitable[JSONResponse]:
        """
        Handles validation errors and returns a custom JSON response.

        Args:
            _ (Request): The incoming HTTP request that caused the validation error.
            exc (RequestValidationError): The validation exception containing error details.

        Returns:
            Response | Awaitable[JSONResponse]: A JSONResponse with error details and HTTP 422 status code.
        """
        # Parse validation errors and organize them into a dictionary
        errors = {}

        for error in exc.errors():
            errors[error["loc"][-1]] = error["msg"]

        response_body = GeneralResponse(
            data="The request contains unprocessable entities.", details=errors
        )

        # Return the response with status 422 and the error details as JSON
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_body.dict(),
        )
