# The gateway's error type and request identifier.
#
# Separate from main.py so that routes can raise a gateway error without
# importing the application module that imports them. Without this split the
# import graph is a cycle, and the usual workaround, importing inside the
# function body, hides the cycle rather than removing it.

from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from app.models import ErrorCode, ErrorDetail, ErrorResponse


class GatewayError(Exception):
    """An error that maps straight onto the documented error envelope.

    Routes raise this rather than building a response, so the envelope is
    constructed in exactly one place and cannot drift between call sites.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def error_response(
    status_code: int,
    code: ErrorCode,
    message: str,
    request_id: str,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    """Render the one error shape the API is documented to return.

    Module level rather than nested in the app factory, because an error can
    arise before the app's handlers run, in a middleware that rejects a request
    outright. Both paths build the envelope here, so it cannot drift between them.
    """
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            retry_after_seconds=retry_after_seconds,
            request_id=request_id,
        )
    )
    headers = {"x-request-id": request_id}
    if retry_after_seconds is not None:
        # The standard header as well as the body field, since HTTP clients and
        # retry libraries read the header and never the body.
        headers["Retry-After"] = str(retry_after_seconds)
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


def request_id(request: Request) -> str:
    """Return a short identifier for this request, generating one if absent.

    Echoed on every response and carried in every error body, so a caller
    reporting a problem can be matched to one request in the logs. An inbound
    header is honoured so that an identifier assigned upstream survives.
    """
    existing = request.headers.get("x-request-id")
    return existing if existing else f"req_{uuid.uuid4().hex[:12]}"
