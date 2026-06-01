"""Central application configuration: one pydantic-settings Settings object.

Reads env (and a local .env), fails fast if a required secret is missing. Module-level `settings`
singleton; Phase 1 moves this behind dependency injection.
"""

from typing import Literal

from cryptography.fernet import Fernet
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )
    # Required (startup fails fast if absent)
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    HUGGINGFACE_TOKEN: str
    AWS_REGION: str
    S3_BUCKET_NAME: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    DATABASE_URL: str  # e.g. postgresql+asyncpg://user:pass@host/db (or postgresql:// — transformed at engine build)

    # Optional
    UPLOADTHING_API_KEY: str | None = None
    PINECONE_INDEX_NAME: str = "rag-knowledge-base"
    LOG_JSON: bool = Field(default=False)
    ENVIRONMENT: Literal["development", "production"] = "development"
    S3_ENDPOINT_URL: str | None = None  # set for MinIO/dev; None → real AWS S3

    # --- Auth (Phase 3) ---
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = 15
    REFRESH_TOKEN_TTL_DAYS: int = 7

    # --- BYOK key encryption (Phase 3) ---
    LLM_KEY_ENCRYPTION_KEY: str  # url-safe base64, 32 bytes — Fernet master key

    # --- CORS (Phase 3) ---
    CORS_ALLOWED_ORIGINS: list[str] = []

    # --- LLM provider (Phase 4) ---
    DEFAULT_LLM_PROVIDER: Literal["gemini", "openai", "anthropic"] = "gemini"
    DEFAULT_LLM_MODEL: str = "gemini-2.5-flash"
    LLM_FALLBACK_API_KEY: SecretStr = SecretStr("")  # optional server-side fallback; BYOK preferred

    # --- Phase 5: Redis / Celery / rate limiting ---
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str | None = None  # falls back to REDIS_URL (see celery_broker_url)
    RATE_LIMIT_STORAGE_URI: str | None = None  # falls back to REDIS_URL; tests set "memory://"
    RATE_LIMIT_CHAT: str = "30/minute"
    RATE_LIMIT_UPLOAD: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "120/minute"

    @property
    def celery_broker_url(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def rate_limit_storage_uri(self) -> str:
        return self.RATE_LIMIT_STORAGE_URI or self.REDIS_URL

    @field_validator("LLM_KEY_ENCRYPTION_KEY")
    @classmethod
    def _validate_fernet_key(cls, v: str) -> str:
        Fernet(v.encode())  # raises ValueError if not a valid 32-byte url-safe base64 key
        return v


settings = Settings()  # raises ValidationError on missing required vars
