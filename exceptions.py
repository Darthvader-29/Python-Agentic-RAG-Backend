from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail


async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


class InvalidTokenTypeError(Exception):
    """Raised when a JWT token has the wrong `type` claim."""

    def __init__(self, expected: str, got: str | None):
        self.expected = expected
        self.got = got
        super().__init__(f"Expected token type '{expected}', got '{got}'")
