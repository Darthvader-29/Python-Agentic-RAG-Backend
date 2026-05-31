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


# ── Phase 4: Provider-neutral LLM error taxonomy ─────────────────────────────


class LLMError(AppException):
    """Base for all provider-neutral LLM failures."""

    status_code = 502
    default_detail = "The AI provider returned an error. Please try again."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(status_code=self.status_code, detail=detail or self.default_detail)


class LLMAuthError(LLMError):
    status_code = 401
    default_detail = "The AI provider rejected the API key. Check the key and permissions."


class LLMRateLimitError(LLMError):
    status_code = 429
    default_detail = "The AI provider rate limit was reached. Please retry later."


class LLMUnavailableError(LLMError):
    status_code = 503
    default_detail = "The AI provider is temporarily unavailable. Please retry later."


class LLMResponseError(LLMError):
    status_code = 502
    default_detail = "The AI provider returned an unusable response."
